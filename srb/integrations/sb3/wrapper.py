from typing import Any, Dict, List, Mapping, Sequence

import gymnasium
import numpy
import torch
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from srb.core.envs import DirectEnv


class Sb3EnvWrapper(VecEnv):
    def __init__(self, env: DirectEnv):
        # Initialize the wrapper
        self.env = env
        # Collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        # Obtain gym spaces
        # Note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        observation_space = self.unwrapped.single_observation_space
        action_space = self.unwrapped.single_action_space
        if isinstance(
            action_space, gymnasium.spaces.Box
        ) and not action_space.is_bounded("both"):
            action_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=action_space.shape
            )

        # Initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        # Add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

        # Use the appropriate observation processing function
        if isinstance(observation_space, gymnasium.spaces.Dict):
            self._process_obs = self._process_obs_dict  # type: ignore

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> DirectEnv:
        return self.env.unwrapped  # type: ignore

    def get_episode_rewards(self) -> Sequence[float]:
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> Sequence[int]:
        return self._ep_len_buf.cpu().tolist()

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs  # type: ignore

    def reset(self) -> VecEnvObs:
        obs_dict, _ = self.env.reset()
        # Reset episodic information buffers
        self._ep_rew_buf.zero_()
        self._ep_len_buf.zero_()
        # Convert data types to numpy depending on backend
        return self._process_obs(obs_dict)  # type: ignore

    def step_async(self, actions):
        # Convert input to numpy array
        if not isinstance(actions, torch.Tensor):
            actions = numpy.asarray(actions)
            actions = torch.from_numpy(actions).to(
                device=self.sim_device, dtype=torch.float32
            )
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # Convert to tensor
        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(
            self._async_actions
        )
        # Update episode un-discounted return and length
        self._ep_rew_buf += rew
        self._ep_len_buf += 1
        # Compute reset ids
        dones = terminated | truncated
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        # Convert data types to numpy depending on backend
        # Note: ManagerBasedRLEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)  # type: ignore
        rew = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        # Convert extra information to list of dicts
        infos = self._process_extras(
            obs,  # type: ignore
            terminated,
            truncated,
            extras,
            reset_ids,  # type: ignore
        )

        # Reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        return obs, rew, dones, infos  # type: ignore

    def close(self):
        self.env.close()

    def get_attr(self, attr_name: str, indices: Sequence[int] | slice | None = None):
        # Resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)  # type: ignore
        # Obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # Return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(
        self, attr_name: str, value: Any, indices: Sequence[int] | None = None
    ):
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Sequence[int] | None = None,
        **method_kwargs,
    ):
        if method_name == "render":
            # Gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # This isn't properly implemented but it is not necessary.
            # Mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError(
            "Checking if environment is wrapped is not supported."
        )

    def get_images(self):
        raise NotImplementedError("Getting images is not supported.")

    def _process_obs(self, obs: torch.Tensor) -> numpy.ndarray:
        return obs.detach().cpu().numpy()

    def _process_obs_dict(
        self, obs: Dict[str, torch.Tensor]
    ) -> Mapping[str, numpy.ndarray]:
        for key, value in obs.items():
            obs[key] = value.detach().cpu().numpy()  # type: ignore
        return obs  # type: ignore

    def _process_extras(
        self,
        obs: numpy.ndarray,
        terminated: numpy.ndarray,
        truncated: numpy.ndarray,
        extras: Dict[str, Any],
        reset_ids: numpy.ndarray,
    ) -> Sequence[Mapping[str, Any]]:
        # Create empty list of dictionaries to fill
        infos: List[Dict[str, Any]] = [
            dict.fromkeys(extras.keys()) for _ in range(self.num_envs)
        ]
        # Fill-in information for each sub-environment
        # Note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # Fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = {}
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # Fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # Fill-in information from extras
            for key, value in extras.items():
                # 1. remap extra episodes information safely
                # 2. for others just store their values
                if key == "log":
                    # Only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # Add information about terminal observation separately
            if idx in reset_ids:
                # Extract terminal observations
                if isinstance(obs, Dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # Add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        # Return list of dictionaries
        return infos
