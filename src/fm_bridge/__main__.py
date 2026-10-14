"""Command line: `fm-bridge [--port N]`."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from aiohttp import web

from .bridge import Bridge
from .server import create_app

log = logging.getLogger("fm_bridge")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fm-bridge",
        description="OpenAI-compatible server for Apple's on-device Foundation Model.",
    )
    parser.add_argument("--host", default=os.environ.get("FM_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FM_BRIDGE_PORT", "1977")))
    parser.add_argument("--tool-timeout", type=float, default=600, metavar="SECONDS",
                        help="how long a generation waits for the client's tool results (default 600)")
    parser.add_argument("--idle-ttl", type=float, default=1800, metavar="SECONDS",
                        help="how long an answered conversation stays cached (default 1800)")
    parser.add_argument("--max-idle", type=int, default=8, metavar="N",
                        help="answered conversations kept for reuse (default 8)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        level=args.log_level.upper())
    bridge = Bridge(tool_timeout=args.tool_timeout, idle_ttl=args.idle_ttl, max_idle=args.max_idle)
    available, reason = bridge.model.is_available()
    if not available:
        sys.exit(f"fm-bridge: the on-device model is unavailable ({reason}). "
                 "Check that Apple Intelligence is turned on.")
    log.info("serving http://%s:%d/v1 (model 'system', %d-token context)",
             args.host, args.port, bridge.model.context_size)
    web.run_app(create_app(bridge), host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
