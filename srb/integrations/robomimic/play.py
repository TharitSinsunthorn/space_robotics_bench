#!/root/isaac-sim/python.sh


import os
import sys

from omni.isaac.lab.app import AppLauncher

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from _cli_utils import add_default_cli_args, argparse, launch_app, shutdown_app


def main(launcher: AppLauncher, args: argparse.Namespace):
    import gymnasium
    import robomimic.utils.fileutils as FileUtils
    import robomimic.utils.torchutils as TorchUtils
    import torch
    from omni.isaac.kit import SimulationApp

    import srb  # noqa: F401
    from srb.utils.cfg import parse_task_cfg

    ## Extract simulation app
    sim_app: SimulationApp = launcher.app

    # parse configuration
    task_cfg = parse_task_cfg(
        args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric
    )
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    task_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gymnasium.make(args.task, cfg=task_cfg)

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=True
    )

    # reset environment
    obs_dict, _ = env.reset()
    # robomimic only cares about policy observations
    obs = obs_dict["policy"]
    # simulate environment
    while sim_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            actions = policy(obs)
            actions = (
                torch.from_numpy(actions)
                .to(device=device)
                .view(1, env.action_space.shape[1])
            )
            # apply actions
            obs_dict = env.step(actions)[0]
            # robomimic only cares about policy observations
            obs = obs_dict["policy"]

    # close the simulator
    env.close()


### Helper functions ###
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_default_cli_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_cli_args()

    # Launch the app
    launcher = launch_app(args)

    # Run the main function
    main(launcher=launcher, args=args)

    # Shutdown the app
    shutdown_app(launcher)
