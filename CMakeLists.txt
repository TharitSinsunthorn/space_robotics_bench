cmake_minimum_required(VERSION 3.5)
project(space_robotics_bench)

## Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

## Install Python package
ament_python_install_package(space_robotics_bench)

## Install Python scripts
set(SCRIPTS_DIR scripts)
install(PROGRAMS
    ${SCRIPTS_DIR}/utils/clean_procgen_cache.py
    ${SCRIPTS_DIR}/utils/tensorboard.bash
    ${SCRIPTS_DIR}/utils/update_assets.bash
    ${SCRIPTS_DIR}/gui.bash
    ${SCRIPTS_DIR}/list_envs.py
    ${SCRIPTS_DIR}/random_agent.py
    ${SCRIPTS_DIR}/ros2.py
    ${SCRIPTS_DIR}/teleop.py
    ${SCRIPTS_DIR}/zero_agent.py
    DESTINATION lib/${PROJECT_NAME}
)

## Install directories
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

## Setup the project
ament_package()
