#!/usr/bin/env bash
### Run the GUI application
### Usage: gui.bash [--release] [CARGO_RUN_ARGS]
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"
CARGO_MANIFEST_PATH="${REPOSITORY_DIR}/Cargo.toml"

## Config
# Options for running the application
RUN_OPTS="${RUN_OPTS:-}"
# Default to release build
WITH_RELEASE_BUILD="${WITH_RELEASE_BUILD:-true}"
if [[ "${WITH_RELEASE_BUILD,,}" = true ]]; then
    RUN_OPTS+=" --release"
fi

## Run with Cargo
CARGO_RUN_CMD=(
    cargo run
    --manifest-path "${CARGO_MANIFEST_PATH}"
    --package srb_gui
    --bin gui
    "${RUN_OPTS}"
    "${*:1}"
)
echo -e "\033[1;90m[TRACE] ${CARGO_RUN_CMD[*]}\033[0m" | xargs
# shellcheck disable=SC2048
exec ${CARGO_RUN_CMD[*]}
