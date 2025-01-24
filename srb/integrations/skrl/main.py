from pathlib import Path
from typing import Literal

from omni.isaac.kit import SimulationApp
from skrl.utils.runner.torch import Runner

from srb.core.envs import DirectEnv
from srb.integrations.skrl.wrapper import SkrlEnvWrapper
from srb.utils.cfg import create_logdir_path, get_last_file, get_last_run_logdir_path

FRAMEWORK_NAME = "skrl"


def run(
    workflow: Literal["train", "eval"],
    env: DirectEnv,
    sim_app: SimulationApp,
    env_id: str,
    env_cfg: dict,
    agent_cfg: dict,
    model: str,
    continue_training: bool | None = None,
    **kwargs,
):
    # Determine logdir and checkpoint path
    match workflow:
        case "train":
            assert not (continue_training and model)
            if continue_training:
                logdir = Path(get_last_run_logdir_path(FRAMEWORK_NAME, env_id))
                from_checkpoint = get_last_file(logdir.joinpath("checkpoints"))
            elif model:
                from_checkpoint = model
                logdir = Path(from_checkpoint).parent.parent
            else:
                logdir = Path(create_logdir_path(FRAMEWORK_NAME, env_id))
                from_checkpoint = ""
        case "eval":
            if model:
                from_checkpoint = model
                logdir = Path(from_checkpoint).parent.joinpath("eval")
            else:
                logdir = Path(get_last_run_logdir_path(FRAMEWORK_NAME, env_id))
                from_checkpoint = get_last_file(logdir.joinpath("checkpoints"))
                logdir = from_checkpoint.parent.joinpath("eval")

    # Update agent config
    agent_cfg["agent"]["experiment"]["directory"] = logdir.parent
    agent_cfg["agent"]["experiment"]["experiment_name"] = logdir

    # Wrap the environment
    env = SkrlEnvWrapper(env)  # type: ignore

    # Create the runner
    runner = Runner(
        env,  # type: ignore
        agent_cfg,
    )

    # Load checkpoint if needed
    if from_checkpoint:
        runner.agent.load(
            from_checkpoint,  # type: ignore
        )

    # Run the workflow
    runner.run(mode=workflow)
