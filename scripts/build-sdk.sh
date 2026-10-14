#!/usr/bin/env bash
# Build Apple's Foundation Models SDK for Python (apple-fm-sdk) as a wheel in vendor/.
#
# PyPI ships only the source package, and its build backend refuses to run without a full
# Xcode install. The Command Line Tools' macOS 27 SDK includes FoundationModels and compiles
# the bindings fine, so this runs the backend's own steps (swift build, ctypesgen, the
# library-path fix-up) and packages the result with plain setuptools. Apple's code is used
# unmodified; only the Xcode check is skipped.
#
#   scripts/build-sdk.sh            # builds the pinned tag
#   SDK_TAG=v0.2.1 scripts/build-sdk.sh

set -euo pipefail

SDK_TAG="${SDK_TAG:-v0.2.1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> fetching apple/python-apple-fm-sdk $SDK_TAG"
git clone --quiet --depth 1 --branch "$SDK_TAG" https://github.com/apple/python-apple-fm-sdk "$WORK/sdk"
cd "$WORK/sdk"

echo "==> compiling the Swift bindings"
swift_args=()
# `Attachment` (image prompts) exists only in the macOS 27+ SDK.
if [[ "$(xcrun --sdk macosx --show-sdk-version | cut -d. -f1)" -ge 27 ]]; then
  swift_args+=(-Xswiftc -DFM_HAS_MACOS_27_SDK)
fi
(cd foundation-models-c && swift build -c release "${swift_args[@]}" >/dev/null)
cp -R "$(cd foundation-models-c && swift build -c release --show-bin-path)" src/apple_fm_sdk/lib

echo "==> generating ctypes bindings and packaging"
mkdir -p "$ROOT/vendor"
uv run --quiet --isolated --no-project --python 3.13 \
  --with ctypesgen --with setuptools --with wheel \
  python - "$ROOT/vendor" <<'EOF'
import pathlib, re, subprocess, sys
sys.path.insert(0, ".")
import build_backend  # the SDK's backend; only its path fix-up helper is used

out = pathlib.Path("src/apple_fm_sdk/_ctypes_bindings.py")
subprocess.run(
    ["ctypesgen", "foundation-models-c/Sources/FoundationModelsCBindings/include/FoundationModels.h",
     "-L", "lib", "-l", "FoundationModels", "-o", str(out)],
    check=True, capture_output=True,
)
code = out.read_text()
if "import os" not in code:
    code = code.replace("import sys\n", "import sys\nimport os\n")
code = re.sub(r"add_library_search_dirs\(\[(.*?)\]\)",
              lambda m: build_backend._fix_library_search_dirs(m.group(1)), code)
out.write_text(code)

from setuptools import build_meta
print("built", build_meta.build_wheel(sys.argv[1]))
EOF

# Every build hashes differently, so point an existing lock at the new wheel.
if [[ -f "$ROOT/uv.lock" ]]; then
  (cd "$ROOT" && uv lock --quiet --upgrade-package apple-fm-sdk)
fi
echo "==> done; now run: uv sync"
