from os import path

# TODO: Convert everything into pathlib.Path objects

# Path to repository root directory
SRB_DIR = path.dirname(path.dirname(path.realpath(__file__)))

# Path to assets directory
SRB_ASSETS_DIR = path.join(SRB_DIR, "assets")
SRB_ASSETS_DIR_SRB = path.join(SRB_ASSETS_DIR, "srb_assets")
SRB_ASSETS_DIR_SRB_HDRI = path.join(SRB_ASSETS_DIR_SRB, "hdri")
SRB_ASSETS_DIR_SRB_MODEL = path.join(SRB_ASSETS_DIR_SRB, "model")
SRB_ASSETS_DIR_SRB_OBJECT = path.join(SRB_ASSETS_DIR_SRB_MODEL, "object")
SRB_ASSETS_DIR_SRB_ROBOT = path.join(SRB_ASSETS_DIR_SRB_MODEL, "robot")
SRB_ASSETS_DIR_SRB_TERRAIN = path.join(SRB_ASSETS_DIR_SRB_MODEL, "terrain")
SRB_ASSETS_DIR_SRB_VEHICLE = path.join(SRB_ASSETS_DIR_SRB_MODEL, "vehicle")

# Path to hyperparameters directory
SRB_HYPERPARAMS_DIR = path.join(SRB_DIR, "hyperparams")

# Path to scripts directory
SRB_SCRIPTS_DIR = path.join(SRB_DIR, "scripts")
