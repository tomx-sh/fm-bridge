# fm-bridge

An OpenAI-compatible Chat Completions server for Apple's on-device Foundation Model, built on
Apple's [Foundation Models SDK for Python](https://github.com/apple/python-apple-fm-sdk).
Point any OpenAI client at `http://127.0.0.1:1977/v1`, model `system`.

Tool calling is native. The framework puts tool definitions into the prompt and runs the tool
loop, so the model decides on its own whether to answer or to call a tool. Nothing is forced
and no escape tool is needed.

## Requirements

- macOS 27 with Apple Intelligence turned on (tested on 27.0.1; the SDK also supports macOS 26,
  without image input)
- Command Line Tools with Swift 6.2+ (a full Xcode is not needed, see below)
- [uv](https://docs.astral.sh/uv/)

## Set up and run

```bash
scripts/build-sdk.sh   # builds Apple's SDK into vendor/ (about 20 s)
uv sync
uv run fm-bridge       # http://127.0.0.1:1977/v1
```

PyPI only ships the SDK as source, and its build backend refuses to run without a full Xcode.
The Command Line Tools' macOS SDK includes FoundationModels and compiles the bindings fine.
`build-sdk.sh` runs the backend's own steps (`swift build`, `ctypesgen`) and packages the
result with plain setuptools. Apple's code is used unmodified. Rerun it after a macOS update.

`uv.lock` is not committed: it pins the hash of the wheel, which differs on every build. The
script refreshes the local lock after each build.

```
fm-bridge [--host 127.0.0.1] [--port 1977] [--tool-timeout 600] [--idle-ttl 1800]
          [--max-idle 8] [--log-level info]
```

`FM_BRIDGE_HOST` and `FM_BRIDGE_PORT` set the defaults for `--host` and `--port`.

To have `fm-bridge` on your `PATH`, run `uv tool install --editable .`. Being editable, it
follows changes to the checkout.

## Use with pi

[pi-fm-bridge](https://github.com/tomx-sh/pi-fm-bridge) adds the model to
[pi](https://pi.dev) as `apple-fm/system`, and starts and stops the bridge for you:

```bash
uv tool install --editable .
pi install git:github.com/tomx-sh/pi-fm-bridge
```

## How tool calls work

The framework wants to run tools itself; OpenAI clients want to run them. The bridge connects
the two:

1. Every tool in the request becomes an `fm.Tool`. Its `call()` runs nothing: it parks the
   call and the server returns it to the client as `tool_calls`, `finish_reason: "tool_calls"`.
   Parallel calls are collected into one response.
2. The client runs the tools and sends the results in its next request. The bridge matches
   them by tool-call id, completes the parked `call()`s, and the **same generation** continues.
   The model sees calls and results in the format it was trained on.
3. When the model answers, the session stays cached. A follow-up request whose history matches
   continues that session instead of starting over.

The bridge rebuilds a conversation it does not hold from the request's messages, through a
Foundation Models transcript. That covers a server restart, eviction, history edited by the
client (compaction, forks), and turns written by another model.

Tool parameters are JSON Schema; the framework accepts a subset. Strings, integers, numbers,
booleans, enums, numeric bounds, item counts, nested objects and arrays of objects map
natively. Boolean arrays, nested arrays, free-form objects, and recursive definitions go out as
JSON strings and are converted back before the client sees the arguments.

## Behaviour and limits

| | |
|---|---|
| Context window | 8192 tokens on macOS 27 (reported by `/v1/models` and `/health`). |
| Context overflow | `400`, `code: "context_length_exceeded"`. |
| Streaming | Real token streaming; `stream_options.include_usage` adds a usage chunk. |
| Usage | Counted with the framework's tokenizer over the whole transcript. |
| Images | Base64 `data:` URLs in user messages. Earlier images in a rebuilt conversation become `[image]`. |
| Sampling | `temperature`, `top_p`, `seed`, `max_tokens` / `max_completion_tokens`. |
| `tool_choice` | `auto` and `none`. The framework cannot be forced to call a tool, so `required` and named tools return `400`. |
| Not supported | `n > 1`, `stop`, `response_format` other than text. Each returns `400`. |
| Guardrails | A safety stop ends the response with `finish_reason: "content_filter"`. |
| Disconnects | A client that disconnects mid-stream cancels its generation. |
| Tool timeout | A generation waits `--tool-timeout` seconds for tool results, then is cancelled. A late result is still served, through a rebuild. |

## Endpoints

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`: model availability, context window, cached conversations by state

## Tests

```bash
uv run pytest            # unit tests, plus end-to-end tests against the model
uv run pytest -m "not live"
```

The live tests skip when the model is unavailable.

## Layout

| Path | Contents |
|---|---|
| `src/fm_bridge/bridge.py` | Conversations, the tool-call bridge, routing and cleanup |
| `src/fm_bridge/server.py` | HTTP layer: request validation, responses, streaming, errors |
| `src/fm_bridge/messages.py` | OpenAI messages to turns, fingerprints and transcript entries |
| `src/fm_bridge/schema.py` | JSON Schema to generation schemas, and arguments back |
| `scripts/build-sdk.sh` | Builds the SDK wheel without Xcode |

## License

[MIT](LICENSE). Apple's Foundation Models SDK, which `build-sdk.sh` downloads and builds, is
licensed separately under Apache-2.0.
