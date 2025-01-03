from pathlib import Path

# TODO: Move to utils

# Path to repository root directory
SRB_DIR = Path(__file__).resolve().parent.parent.parent

# Path to apps (experience) directory
SRB_APPS_DIR = SRB_DIR.joinpath("apps")

# Path to assets directory
SRB_ASSETS_DIR = SRB_DIR.joinpath("assets")
SRB_ASSETS_DIR_SRB = SRB_ASSETS_DIR.joinpath("srb_assets")
SRB_ASSETS_DIR_SRB_HDRI = SRB_ASSETS_DIR_SRB.joinpath("hdri")
SRB_ASSETS_DIR_SRB_MODEL = SRB_ASSETS_DIR_SRB.joinpath("model")
SRB_ASSETS_DIR_SRB_OBJECT = SRB_ASSETS_DIR_SRB_MODEL.joinpath("object")
SRB_ASSETS_DIR_SRB_ROBOT = SRB_ASSETS_DIR_SRB_MODEL.joinpath("robot")
SRB_ASSETS_DIR_SRB_TERRAIN = SRB_ASSETS_DIR_SRB_MODEL.joinpath("terrain")
SRB_ASSETS_DIR_SRB_VEHICLE = SRB_ASSETS_DIR_SRB_MODEL.joinpath("vehicle")

# Path to hyperparameters directory
SRB_HYPERPARAMS_DIR = SRB_DIR.joinpath("hyperparams")

# Path to scripts directory
SRB_SCRIPTS_DIR = SRB_DIR.joinpath("scripts")
