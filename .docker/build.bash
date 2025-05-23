#!/usr/bin/env bash
### Build the Docker image
### Usage: build.bash [TAG] [BUILD_ARGS...]
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"

## If the current user is not in the docker group, all docker commands will be run as root
if ! grep -qi /etc/group -e "docker.*${USER}"; then
    echo "[INFO] The current user '${USER}' is not detected in the docker group. All docker commands will be run as root."
    WITH_SUDO="sudo"
fi

## Determine the name of the image to build
DOCKERHUB_USER="$(${WITH_SUDO} docker info 2>/dev/null | sed '/Username:/!d;s/.* //')"
PROJECT_NAME="$(basename "${REPOSITORY_DIR}")"
IMAGE_NAME="${DOCKERHUB_USER:+${DOCKERHUB_USER}/}${PROJECT_NAME,,}"

## Parse TAG and forward additional build arguments
if [ "${#}" -gt "0" ]; then
    if [[ "${1}" != "-"* ]]; then
        IMAGE_NAME+=":${1}"
        BUILD_ARGS=${*:2}
    else
        BUILD_ARGS=${*:1}
    fi
fi

## Initialize assets if necessary
if [ "$(find "${REPOSITORY_DIR}/assets/srb_assets/" -mindepth 1 -maxdepth 1 | wc -l)" -eq 0 ]; then
    echo "[INFO] Initializing asset submodules"
    if ! "${REPOSITORY_DIR}/assets/update.bash"; then
        echo >&2 -e "\033[1;31m[ERROR] Failed to initialize asset submodules.\033[0m"
    fi
fi

## Build the image
DOCKER_BUILD_CMD=(
    "${WITH_SUDO}" docker build
    "${REPOSITORY_DIR}"
    --file "${REPOSITORY_DIR}/Dockerfile"
    --tag "${IMAGE_NAME}"
    "${BUILD_ARGS}"
)
echo -e "\033[1;90m[TRACE] ${DOCKER_BUILD_CMD[*]}\033[0m" | xargs
# shellcheck disable=SC2048
exec ${DOCKER_BUILD_CMD[*]}
