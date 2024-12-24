from pydantic import BaseModel

from space_robotics_bench.core.asset import AssetBaseCfg


class AssetCfg(BaseModel):
    ## Model
    asset_cfg: AssetBaseCfg
