from space_robotics_bench.utils import register_srb_tasks

from .task import Task, TaskCfg

BASE_TASK_NAME = __name__.split(".")[-1]
register_srb_tasks(
    {
        BASE_TASK_NAME: {},
    },
    default_entry_point=Task,
    default_task_cfg=TaskCfg,
)
