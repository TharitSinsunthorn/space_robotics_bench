from typing import Dict, Literal, TypedDict

from torch import Tensor


class IntermediateTaskState(TypedDict):
    obs: Dict[
        str | Literal["state", "state_dyn", "proprio", "proprio_dyn", "command"],
        Dict[str, Tensor],
    ]
    rew: Dict[str, Tensor]
    term: Tensor
    trunc: Tensor
