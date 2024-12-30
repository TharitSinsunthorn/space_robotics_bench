from typing import Annotated

from pydantic import BaseModel

from space_robotics_bench.core.envs.env_cfg.asset import Assets
from space_robotics_bench.core.envs.env_cfg.domain import Domain
from space_robotics_bench.utils.typing import EnumNameSerializer


class EnvironmentConfig(BaseModel):
    domain: Annotated[Domain, EnumNameSerializer] = Domain.MOON
    assets: Assets = Assets()
