from pathlib import Path
from typing import Literal

import gymnasium
import torch
from omni.isaac.kit import SimulationApp
from skrl.utils.runner.torch import Runner

from srb.core.envs import BaseEnv
from srb.integrations.skrl.wrapper import SkrlEnvWrapper
from srb.utils.parsing import (
    create_logdir_path,
    get_last_file,
    get_last_run_logdir_path,
)

FRAMEWORK_NAME = "skrl"


def run(
    workflow: Literal["train", "eval"],
    env: BaseEnv,
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
    env = NormalizeReward(env)  # type: ignore
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


class NormalizeReward(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gymnasium.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, gamma=gamma, epsilon=epsilon
        )
        gymnasium.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.return_rms = RunningMeanStd(shape=(), device=self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action: torch.Tensor):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        self.returns = (
            self.returns
            * self.gamma
            * (1 - terminateds.to(dtype=torch.float32, device=self.device))
            + rews
        )
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / torch.sqrt(self.return_rms.var + self.epsilon)


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
