from typing import Annotated

from pydantic import BaseModel

from srb.core.envs.env_cfg.asset import Assets
from srb.core.envs.env_cfg.domain import Domain
from srb.utils.typing import EnumNameSerializer


class EnvironmentConfig(BaseModel):
    domain: Annotated[Domain, EnumNameSerializer] = Domain.MOON
    assets: Assets = Assets()
    seed: int = 0
