## Base <https://hub.docker.com/_/ubuntu>
ARG BASE_IMAGE_NAME="ubuntu"
ARG BASE_IMAGE_TAG="jammy"

## Isaac Sim <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim>
## Label as isaac-sim for copying into the final image
ARG ISAAC_SIM_IMAGE_NAME="nvcr.io/nvidia/isaac-sim"
ARG ISAAC_SIM_IMAGE_TAG="4.2.0"
FROM ${ISAAC_SIM_IMAGE_NAME}:${ISAAC_SIM_IMAGE_TAG} AS isaac-sim

## Continue with the base image
FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}

## Use bash as the default shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

## Create a barebones entrypoint that is conditionally updated throughout the Dockerfile
RUN echo "#!/usr/bin/env bash" >> /entrypoint.bash && \
    chmod +x /entrypoint.bash

## Copy Isaac Sim into the base image
ARG ISAAC_SIM_PATH="/root/isaac-sim"
ENV ISAAC_SIM_PYTHON="${ISAAC_SIM_PATH}/python.sh"
COPY --from=isaac-sim /isaac-sim "${ISAAC_SIM_PATH}"
COPY --from=isaac-sim /root/.nvidia-omniverse/config /root/.nvidia-omniverse/config
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
RUN ISAAC_SIM_VERSION="$(cut -d'-' -f1 < "${ISAAC_SIM_PATH}/VERSION")" && \
    echo -e "\n# Isaac Sim ${ISAAC_SIM_VERSION}" >> /entrypoint.bash && \
    echo "export ISAAC_SIM_PATH=\"${ISAAC_SIM_PATH}\"" >> /entrypoint.bash && \
    echo "export OMNI_KIT_ALLOW_ROOT=\"1\"" >> /entrypoint.bash
## Fix cosmetic issues in `isaac-sim/setup_python_env.sh` that append nonsense paths to `PYTHONPATH` and `LD_LIBRARY_PATH`
# hadolint ignore=SC2016
RUN sed -i 's|$SCRIPT_DIR/../../../$LD_LIBRARY_PATH:||' "${ISAAC_SIM_PATH}/setup_python_env.sh" && \
    sed -i 's|$SCRIPT_DIR/../../../$PYTHONPATH:||' "${ISAAC_SIM_PATH}/setup_python_env.sh"

## Optimization: Build Python to improve the runtime performance of training
ARG PYTHON_VERSION="3.10.15"
ENV PYTHONEXE="/usr/local/bin/python${PYTHON_VERSION%%.*}"
# hadolint ignore=DL3003,DL3008
RUN PYTHON_DL_PATH="/tmp/Python-${PYTHON_VERSION}.tar.xz" && \
    PYTHON_SRC_DIR="/tmp/python${PYTHON_VERSION}" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libbz2-dev \
    libdb4o-cil-dev \
    libgdm-dev \
    libhidapi-dev \
    liblzma-dev \
    libncurses5-dev \
    libpcap-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libtk8.6 \
    lzma \
    xz-utils && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto "=https" --tlsv1.2 -sSfL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" -o "${PYTHON_DL_PATH}" && \
    mkdir -p "${PYTHON_SRC_DIR}" && \
    tar xf "${PYTHON_DL_PATH}" -C "${PYTHON_SRC_DIR}" --strip-components=1 && \
    rm "${PYTHON_DL_PATH}" && \
    cd "${PYTHON_SRC_DIR}" && \
    "${PYTHON_SRC_DIR}/configure" --enable-shared --enable-optimizations --with-lto --prefix="/usr/local" && \
    make -j "$(nproc)" && \
    make install && \
    ln -sr "${PYTHONEXE}" /usr/local/bin/python && \
    cd - && \
    rm -rf "${PYTHON_SRC_DIR}"
## Fix `PYTHONEXE` by disabling the append of "isaac-sim/kit/kernel/plugins" to `LD_LIBRARY_PATH` inside `isaac-sim/setup_python_env.sh`
# hadolint ignore=SC2016
RUN sed -i 's|$SCRIPT_DIR/kit/kernel/plugins:||' "${ISAAC_SIM_PATH}/setup_python_env.sh"

## Install system dependencies
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    # Common
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    mold \
    xz-utils \
    # Isaac Sim
    libgl1 \
    libglu1 \
    libxt-dev \
    # Blender
    libegl1 \
    # egui
    libxkbcommon-x11-0 \
    # r2r
    clang \
    # Video recording/processing
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

## Upgrade pip
RUN "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --upgrade pip

## Install Rust
ARG RUST_VERSION="1.83"
RUN echo -e "\n# Rust ${RUST_VERSION}" >> /entrypoint.bash && \
    echo "export PATH=\"${HOME}/.cargo/bin\${PATH:+:\${PATH}}\"" >> /entrypoint.bash && \
    echo "export CARGO_TARGET_DIR=\"${HOME}/.cargo/target\"" >> /entrypoint.bash && \
    echo "export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS=\"-Clink-arg=-fuse-ld=mold -Ctarget-cpu=native\"" >> /entrypoint.bash && \
    echo -e "\n# PyO3" >> /entrypoint.bash && \
    echo "export PYO3_PYTHON=\"${ISAAC_SIM_PYTHON}\"" >> /entrypoint.bash && \
    curl --proto "=https" --tlsv1.2 -sSfL "https://sh.rustup.rs" | sh -s -- --no-modify-path -y --default-toolchain "${RUST_VERSION}" --profile default

## Install (Space) ROS
ARG INSTALL_SPACEROS=false
ARG ROS_DISTRO="humble"
# hadolint ignore=SC1091,DL3008
RUN if [[ "${INSTALL_SPACEROS,,}" != true ]]; then \
    curl --proto "=https" --tlsv1.2 -sSfL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo "${UBUNTU_CODENAME}") main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    ros-dev-tools \
    "ros-${ROS_DISTRO}-ros-base" \
    "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" && \
    rm -rf /var/lib/apt/lists/* && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir catkin_pkg && \
    rosdep init --rosdistro "${ROS_DISTRO}" && \
    echo -e "\n# ROS ${ROS_DISTRO^}" >> /entrypoint.bash && \
    echo "source \"/opt/ros/${ROS_DISTRO}/setup.bash\" --" >> /entrypoint.bash ; \
    fi
ARG SPACEROS_TAG="${ROS_DISTRO}-2024.10.0"
# hadolint ignore=SC1091,DL3003,DL3008
RUN if [[ "${INSTALL_SPACEROS,,}" = true ]]; then \
    curl --proto "=https" --tlsv1.2 -sSfL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo "${UBUNTU_CODENAME}") main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    clang-14 \
    cmake \
    g++ \
    gcc \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-test-dev \
    libboost-thread-dev \
    libedit-dev \
    libgmp-dev \
    libsqlite3-dev \
    libtbb-dev \
    libz-dev \
    llvm-14 \
    llvm-14-dev \
    llvm-14-tools \
    ros-dev-tools && \
    rm -rf /var/lib/apt/lists/* && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir catkin_pkg rosinstall_generator && \
    rosdep init --rosdistro "${ROS_DISTRO}" && \
    SPACEROS_DIR="/opt/spaceros" && \
    SPACEROS_SRC_DIR="${SPACEROS_DIR}/src" && \
    SPACEROS_INSTALL_DIR="${SPACEROS_DIR}/install" && \
    SPACEROS_LOG_DIR="${SPACEROS_DIR}/log" && \
    mkdir -p "${SPACEROS_SRC_DIR}" && \
    vcs import --shallow --recursive --force --input "https://raw.githubusercontent.com/space-ros/space-ros/refs/tags/${SPACEROS_TAG}/ros2.repos" "${SPACEROS_SRC_DIR}" && \
    vcs import --shallow --recursive --force --input "https://raw.githubusercontent.com/space-ros/space-ros/refs/tags/${SPACEROS_TAG}/ikos.repos" "${SPACEROS_SRC_DIR}" && \
    vcs import --shallow --recursive --force --input "https://raw.githubusercontent.com/space-ros/space-ros/refs/tags/${SPACEROS_TAG}/spaceros.repos" "${SPACEROS_SRC_DIR}" && \
    apt-get update && \
    rosdep update --rosdistro "${ROS_DISTRO}" && \
    DEBIAN_FRONTEND=noninteractive rosdep install --default-yes --ignore-src -r --rosdistro "${ROS_DISTRO}" --from-paths "${SPACEROS_SRC_DIR}" --skip-keys "$(curl --proto =https --tlsv1.2 -sSfL https://raw.githubusercontent.com/space-ros/space-ros/refs/tags/${SPACEROS_TAG}/excluded-pkgs.txt | tr '\n' ' ') urdfdom_headers ikos" && \
    rm -rf /var/lib/apt/lists/* /root/.ros/rosdep/sources.cache && \
    cd "${SPACEROS_DIR}" && \
    colcon build --cmake-args -DPython3_EXECUTABLE="${ISAAC_SIM_PYTHON}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_CONFIG_EXECUTABLE="/usr/lib/llvm-14/bin/llvm-config" --no-warn-unused-cli && \
    cd - && \
    rm -rf "${SPACEROS_LOG_DIR}" && \
    echo -e "\n# Space ROS ${ROS_DISTRO^}" >> /entrypoint.bash && \
    echo "source \"${SPACEROS_INSTALL_DIR}/setup.bash\" --" >> /entrypoint.bash ; \
    fi

## Install Blender
ARG BLENDER_PATH="/root/blender"
ARG BLENDER_VERSION="4.3.0"
# hadolint ignore=SC2016
RUN echo -e "\n# Blender ${BLENDER_VERSION}" >> /entrypoint.bash && \
    echo "export PATH=\"${BLENDER_PATH}\${PATH:+:\${PATH}}\"" >> /entrypoint.bash && \
    curl --proto "=https" --tlsv1.2 -sSfL "https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" -o "/tmp/blender_${BLENDER_VERSION}.tar.xz" && \
    mkdir -p "${BLENDER_PATH}" && \
    tar xf "/tmp/blender_${BLENDER_VERSION}.tar.xz" -C "${BLENDER_PATH}" --strip-components=1 && \
    rm "/tmp/blender_${BLENDER_VERSION}.tar.xz"

## Install Isaac Lab
ARG ISAACLAB_PATH="/root/isaaclab"
ARG ISAACLAB_REMOTE="https://github.com/isaac-sim/IsaacLab.git"
ARG ISAACLAB_BRANCH="main"
ARG ISAACLAB_COMMIT_SHA="dd044a4977722d85e966c5b63949093c9b14667f" # 2024-12-20
# hadolint ignore=SC2044
RUN echo -e "\n# Isaac Lab ${ISAACLAB_COMMIT_SHA}" >> /entrypoint.bash && \
    echo "export ISAACLAB_PATH=\"${ISAACLAB_PATH}\"" >> /entrypoint.bash && \
    git clone "${ISAACLAB_REMOTE}" "${ISAACLAB_PATH}" --branch "${ISAACLAB_BRANCH}" && \
    git -C "${ISAACLAB_PATH}" reset --hard "${ISAACLAB_COMMIT_SHA}" && \
    for extension in $(find -L "${ISAACLAB_PATH}/source/extensions" -mindepth 1 -maxdepth 1 -type d); do \
    if [ -f "${extension}/setup.py" ]; then \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --editable "${extension}" ; \
    fi ; \
    done

## Define the workspace of the project
ARG SRB_PATH="/root/ws"
RUN echo -e "\n# Space Robotics Bench" >> /entrypoint.bash && \
    echo "export SRB_PATH=\"${SRB_PATH}\"" >> /entrypoint.bash
WORKDIR "${SRB_PATH}"

## Finalize the entrypoint
# hadolint ignore=SC2016
RUN echo -e "\n# Execute command" >> /entrypoint.bash && \
    echo -en 'exec "${@}"\n' >> /entrypoint.bash && \
    sed -i '$a source /entrypoint.bash --' ~/.bashrc
ENTRYPOINT ["/entrypoint.bash"]

## Install dependencies
# hadolint ignore=DL3013,SC2046
RUN --mount=type=bind,source=pyproject.toml,target="${SRB_PATH}/pyproject.toml" \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --ignore-installed toml~=0.10 && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir $(python -c "f='${SRB_PATH}/pyproject.toml'; from toml import load; print(' '.join(filter(lambda d: not d.startswith(p['name']), (*p.get('dependencies', ()), *(d for ds in p.get('optional-dependencies', {}).values() for d in ds)))) if (p := load(f).get('project', None)) else '')")

## Install ROS dependencies in advance to cache the layers (speeds up rebuilds)
COPY ./package.xml "${SRB_PATH}/"
RUN apt-get update && \
    rosdep update --rosdistro "${ROS_DISTRO}" && \
    DEBIAN_FRONTEND=noninteractive rosdep install --default-yes --ignore-src --rosdistro "${ROS_DISTRO}" --from-paths "${SRB_PATH}" && \
    rm -rf /var/lib/apt/lists/* /root/.ros/rosdep/sources.cache

## Copy the source code into the image
COPY . "${SRB_PATH}"

# ## Build Rust targets
# # hadolint ignore=SC1091
# RUN source /entrypoint.bash -- && \
#     cargo build --release --workspace --all-targets

## Install project as editable Python module
# hadolint ignore=SC1091
RUN source /entrypoint.bash -- && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --no-deps --editable "${SRB_PATH}[all]"

## Install project as ROS package
ARG ROS_WS="/opt/ros/${ROS_DISTRO}/ws"
# hadolint ignore=SC1091
RUN source /entrypoint.bash -- && \
    colcon build --merge-install --symlink-install --cmake-args -DPython3_EXECUTABLE="${ISAAC_SIM_PYTHON}" --paths "${SRB_PATH}" --build-base "${ROS_WS}/build" --install-base "${ROS_WS}/install" && \
    rm -rf ./log && \
    sed -i "s|source \"/opt/ros/${ROS_DISTRO}/setup.bash\" --|source \"${ROS_WS}/install/setup.bash\" --|" /entrypoint.bash

## Set the default command
CMD ["bash"]

############
### Misc ###
############

## Skip writing Python bytecode to the disk to avoid polluting mounted host volume with `__pycache__` directories
ENV PYTHONDONTWRITEBYTECODE=1

###############
### Develop ###
###############

# ## Install DreamerV3 locally to enable mounting the source code into the container
# ARG DREAMERV3_PATH="/root/dreamerv3"
# ARG DREAMERV3_REMOTE="https://github.com/AndrejOrsula/dreamerv3.git"
# ARG DREAMERV3_BRANCH="dev"
# RUN git clone "${DREAMERV3_REMOTE}" "${DREAMERV3_PATH}" --branch "${DREAMERV3_BRANCH}" && \
#     "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir -r "${DREAMERV3_PATH}/embodied/requirements.txt" && \
#     "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir -r "${DREAMERV3_PATH}/dreamerv3/requirements.txt" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" && \
#     "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --editable "${DREAMERV3_PATH}"
