from pydantic import BaseModel

from space_robotics_bench.core.asset.utils.transform import Transform


class Frame(BaseModel):
    prim_relpath: str
    offset: Transform = Transform()
