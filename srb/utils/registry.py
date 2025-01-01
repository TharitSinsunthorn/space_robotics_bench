from typing import Any, List, Literal, Mapping

import gymnasium

from srb.paths import SRB_HYPERPARAMS_DIR
from srb.utils.cfg import parse_algo_configs

SRB_NAMESPACE: str = "srb"


def register_srb_tasks(
    tasks: Mapping[
        str,
        Mapping[
            Literal["entry_point", "task_cfg", "cfg_dir"],
            gymnasium.Env | Any | str,
        ],
    ],
    *,
    default_entry_point: gymnasium.Env | None = None,
    default_task_cfg: Any | None = None,
    default_cfg_dir: str | None = SRB_HYPERPARAMS_DIR,
):
    for id, cfg in tasks.items():
        entry_point: gymnasium.Env = cfg.get("entry_point", default_entry_point)  # type: ignore
        gymnasium.register(
            id=f"{SRB_NAMESPACE}/{id}",
            entry_point=f"{entry_point.__module__}:{entry_point.__name__}",  # type: ignore
            kwargs={
                "task_cfg": cfg.get("task_cfg", default_task_cfg),
                **parse_algo_configs(cfg.get("cfg_dir", default_cfg_dir)),  # type: ignore
            },
            disable_env_checker=True,
        )


def get_srb_tasks() -> List[str]:
    return [
        env_id
        for env_id in gymnasium.registry.keys()
        if env_id.startswith(f"{SRB_NAMESPACE}/")
    ]
