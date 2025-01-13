import os
from typing import Mapping

SUPPORTED_FRAMEWORKS = {
    "dreamer": {"multi_algo": False},
    "sb3": {"multi_algo": True},
    "robomimic": {"multi_algo": True},
}
SUPPORTED_CFG_FILE_EXTENSIONS = (
    "json",
    "toml",
    "yaml",
    "yml",
)
FRAMEWORK_CFG_ENTRYPOINT_KEY = "{FRAMEWORK}_cfg"
FRAMEWORK_MULTI_ALGO_CFG_ENTRYPOINT_KEY = "{FRAMEWORK}_{ALGO}_cfg"


def parse_algo_configs(cfg_dir: str) -> Mapping[str, str]:
    algo_config = {}

    for root, _, files in os.walk(cfg_dir):
        for file in files:
            if not file.endswith(SUPPORTED_CFG_FILE_EXTENSIONS):
                continue
            file = os.path.join(root, file)

            key = _identify_config(root, file)
            if key is not None:
                algo_config[key] = file

    return algo_config


def _identify_config(root: str, file) -> str | None:
    basename = os.path.basename(file).split(".")[0]

    for framework, properties in SUPPORTED_FRAMEWORKS.items():
        if root.endswith(framework):
            assert properties["multi_algo"]
            if "_" in basename:
                algo = basename.split("_")[0]
            else:
                algo = basename
            return FRAMEWORK_MULTI_ALGO_CFG_ENTRYPOINT_KEY.format(
                FRAMEWORK=framework, ALGO=algo
            )
        elif basename.startswith(f"{framework}"):
            if properties["multi_algo"]:
                algo = basename[len(framework) + 1 :].split("_")[0]
                return FRAMEWORK_MULTI_ALGO_CFG_ENTRYPOINT_KEY.format(
                    FRAMEWORK=framework, ALGO=algo
                )
            else:
                return FRAMEWORK_CFG_ENTRYPOINT_KEY.format(FRAMEWORK=framework)

    return None
