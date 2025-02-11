#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"

## Config
# Additional volumes to mount inside the container
EXTRA_DOCKER_VOLUMES=(
    "${HOME}/Videos:/root/Videos"
)
# Additional environment variables to set inside the container
EXTRA_DOCKER_ENVIRON=()

## Parse arguments
DEFAULT_CMD="srb gui"
if [ "${#}" -gt "0" ]; then
    CMD=${*:1}
fi

## Run the container
DOCKER_RUN_CMD=(
    "${SCRIPT_DIR}/run.bash"
    "${EXTRA_DOCKER_VOLUMES[@]/#/"-v "}"
    "${EXTRA_DOCKER_ENVIRON[@]/#/"-e "}"
    "${CMD:-${DEFAULT_CMD}}"
)
echo -e "\033[1;90m[TRACE] ${DOCKER_RUN_CMD[*]}\033[0m" | xargs
# shellcheck disable=SC2048
exec ${DOCKER_RUN_CMD[*]}
