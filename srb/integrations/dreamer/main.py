from functools import partial as bind
from pathlib import Path
from typing import Any, Dict, Literal

import elements
import embodied
import portal
import ruamel.yaml as yaml
from dreamerv3 import agent as dreamer_agent
from dreamerv3 import main as dreamer_main
from omni.isaac.kit import SimulationApp

from srb.core.envs import BaseEnv
from srb.integrations.dreamer.eval import eval_only
from srb.integrations.dreamer.train import train
from srb.integrations.dreamer.wrapper import EmbodiedEnvWrapper
from srb.utils.parsing import create_logdir_path, get_last_run_logdir_path

ALGO_NAME = "dreamer"
UPSTREAM_CONFIG_PATH = Path(dreamer_agent.__file__).parent.joinpath("configs.yaml")
SAVE_REPLAY: bool = False


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
                logdir = Path(get_last_run_logdir_path(ALGO_NAME, env_id))
                from_checkpoint = logdir.joinpath("ckpt").joinpath(
                    logdir.joinpath("ckpt").joinpath("latest").read_text().strip()
                )
            elif model:
                from_checkpoint = model
                logdir = Path(from_checkpoint).parent.parent
            else:
                logdir = Path(create_logdir_path(ALGO_NAME, env_id))
                from_checkpoint = ""
        case "eval":
            if model:
                from_checkpoint = model
                logdir = Path(from_checkpoint).joinpath("eval")
            else:
                logdir = Path(get_last_run_logdir_path(ALGO_NAME, env_id))
                from_checkpoint = logdir.joinpath("ckpt").joinpath(
                    logdir.joinpath("ckpt").joinpath("latest").read_text().strip()
                )
                logdir = from_checkpoint.joinpath("eval")

    # Setup logdir
    logdir = elements.Path(logdir)
    logdir.mkdir()
    print("Agent logdir:", logdir)

    # Load the config
    configs: Dict[str, Any] = yaml.YAML(typ="safe").load(
        UPSTREAM_CONFIG_PATH.read_text()
    )
    config = elements.Config(configs["defaults"])
    config = config.update(
        {
            **agent_cfg,
            "task": env_id.replace("/", "_"),
            "logdir": logdir,
            "run.from_checkpoint": from_checkpoint,
            "run.envs": env_cfg.scene.num_envs,  # type: ignore
            "run.eval_envs": env_cfg.scene.num_envs,  # type: ignore
        }
    )

    # Save the config
    config.save(logdir / "config.yaml")

    # Wrap the environment
    env = EmbodiedEnvWrapper(env)  # type: ignore
    for name, space in env.act_space.items():  # type: ignore
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)  # type: ignore
    env = embodied.wrappers.UnifyDtypes(env)  # type: ignore
    for name, space in env.act_space.items():  # type: ignore
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)  # type: ignore

    # Setup the workflow
    def init():
        elements.timer.global_timer.enabled = config.logger.timer  # type: ignore

    portal.setup(
        errfile=config.errfile and logdir / "error",
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    def make_env():
        return env

    def make_agent(config):
        obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith("log/")}  # type: ignore
        act_space = {k: v for k, v in env.act_space.items() if k != "reset"}  # type: ignore
        if config.random_agent:
            return embodied.RandomAgent(obs_space, act_space)
        cpdir = elements.Path(config.logdir)
        cpdir = cpdir.parent if config.replicas > 1 else cpdir
        return dreamer_agent.Agent(
            obs_space,
            act_space,
            elements.Config(
                **config.agent,
                logdir=config.logdir,
                seed=config.seed,
                jax=config.jax,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
                replay_context=config.replay_context,
                report_length=config.report_length,
                replica=config.replica,
                replicas=config.replicas,
            ),
        )

    # Run the workflow
    match workflow:
        case "train":
            train(
                bind(make_agent, config),
                bind(
                    dreamer_main.make_replay, config, "replay" if SAVE_REPLAY else None
                ),
                make_env,
                bind(dreamer_main.make_stream, config),
                bind(dreamer_main.make_logger, config),
                args,
                sim_app=sim_app,
            )
        case "eval":
            eval_only(
                bind(make_agent, config),
                make_env,
                bind(dreamer_main.make_logger, config),
                args,
                sim_app=sim_app,
            )
