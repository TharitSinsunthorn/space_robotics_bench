import functools
from typing import Literal

import dreamerv3
import embodied
import gymnasium
import numpy as np
import torch

from space_robotics_bench.core.envs import BaseEnv, BaseEnvManaged


def process_dreamerv3_cfg(
    cfg: dict,
    logdir: str,
    num_envs: int,
    task_name: str = "unknown_task",
    model_size: Literal[
        "debug", "size12m", "size25m", "size50m", "size100m", "size200m", "size400m"
    ] = "debug",
) -> embodied.Config:
    config = embodied.Config(dreamerv3.Agent.configs["defaults"])

    config = config.update(
        dreamerv3.Agent.configs[model_size],
    )
    config = config.update(cfg)
    config = config.update(
        {
            "logdir": logdir,
            "task": task_name,
            "run.num_envs": num_envs,
        }
    )
    config = embodied.Flags(config).parse(argv=[])
    config = config.update(
        replay_length=config.replay_length or config.batch_length,
        replay_length_eval=config.replay_length_eval or config.batch_length_eval,
    )

    return config


class EmbodiedEnvWrapper(embodied.Env):
    def __init__(
        self,
        env: BaseEnv | BaseEnvManaged,
        obs_key="image",
        act_key="action",
    ):
        # check that input is valid
        if not isinstance(env.unwrapped, (BaseEnv, BaseEnvManaged)):
            raise ValueError(
                f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}"
            )
        # initialize the wrapper
        self._env = env
        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode
        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

        self._action_space = self.unwrapped.single_action_space
        if (
            isinstance(self._action_space, gymnasium.spaces.Box)
            and self._action_space.is_bounded() != "both"
        ):
            self._action_space = gymnasium.spaces.Box(
                low=-100.0, high=100.0, shape=self._action_space.shape
            )

        self._obs_dict = hasattr(self.unwrapped.single_observation_space, "spaces")
        self._act_dict = hasattr(self._action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key

        self._done = np.ones(self.num_envs, dtype=bool)
        # self._is_first = np.zeros(self.num_envs, dtype=bool)
        self._info = [None for _ in range(self.num_envs)]

    def __len__(self):
        return self.num_envs

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> BaseEnv | BaseEnvManaged:
        return self.env.unwrapped

    def get_episode_rewards(self) -> list[float]:
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> list[int]:
        return self._ep_len_buf.cpu().tolist()

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self.unwrapped.single_observation_space.spaces)
        else:
            spaces = {self._obs_key: self.unwrapped.single_observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._action_space.spaces)
        else:
            spaces = {self._act_key: self._action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = embodied.Space(bool)
        return spaces

    def _convert_space(self, space):
        if hasattr(space, "n"):
            return embodied.Space(np.int32, (), 0, space.n)
        else:
            return embodied.Space(space.dtype, space.shape, space.low, space.high)

    def seed(self, seed: int | None = None) -> list[int | None]:
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def reset(self):  # noqa: D102
        obs, self._info = self._env.reset()
        self._done = np.zeros(self.num_envs, dtype=bool)
        # self._is_first = np.zeros(self.num_envs, dtype=bool)
        return self._obs(
            obs=obs,
            reward=np.zeros(self.num_envs, dtype=np.float32),
            is_first=np.ones(self.num_envs, dtype=bool),
            is_last=np.zeros(self.num_envs, dtype=bool),
            is_terminal=np.zeros(self.num_envs, dtype=bool),
        )

    def step(self, action):
        if action["reset"].all() or self._done.all():
            return self.reset()

        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]

        if not isinstance(action, torch.Tensor):
            action = np.asarray(action)
            action = torch.from_numpy(action.copy()).to(
                device=self.sim_device, dtype=torch.float32
            )
        else:
            action = action.to(device=self.sim_device, dtype=torch.float32)

        obs, reward, terminated, truncated, self._info = self._env.step(action)

        # update episode un-discounted return and length
        self._ep_rew_buf += reward
        self._ep_len_buf += 1

        self._done = (terminated | truncated).detach().cpu().numpy()
        reset_ids = (self._done > 0).nonzero()

        reward = reward.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()

        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        # is_first = self._is_first.copy()
        # self._is_first = self._done.copy()

        return self._obs(
            obs=obs,
            reward=reward,
            # is_first=is_first,
            is_first=np.zeros(self.num_envs, dtype=bool),
            is_last=self._done,
            is_terminal=terminated,
        )

    def render(self):
        image = self._env.render()
        assert image is not None
        return image

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def get_attr(self, attr_name, indices=None):
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # this isn't properly implemented but it is not necessary.
            # mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def _obs(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        reward: np.ndarray,
        is_first: np.ndarray,
        is_last: np.ndarray,
        is_terminal: np.ndarray,
    ):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: v.detach().cpu().numpy() for k, v in obs.items()}
        obs.update(
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        return obs

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gymnasium.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)


class DriverParallelEnv:
    def __init__(self, env, num_envs: int, **kwargs):
        self.kwargs = kwargs
        self.length = num_envs
        self.env = env
        self.act_space = self.env.act_space
        self.callbacks = []
        self.acts = None
        self.carry = None
        self.reset()

    def reset(self, init_policy=None):
        self.acts = {
            k: np.zeros((self.length,) + v.shape, v.dtype)
            for k, v in self.act_space.items()
        }
        self.acts["reset"] = np.ones(self.length, bool)
        self.carry = init_policy and init_policy(self.length)

    def close(self):
        self.env.close()

    def on_step(self, callback):
        self.callbacks.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        acts = self.acts
        assert all(len(x) == self.length for x in acts.values())
        assert all(isinstance(v, np.ndarray) for v in acts.values())
        obs = self.env.step(acts)
        assert all(len(x) == self.length for x in obs.values()), obs
        acts, outs, self.carry = policy(obs, self.carry, **self.kwargs)
        assert all(k not in acts for k in outs), (list(outs.keys()), list(acts.keys()))
        if obs["is_last"].any():
            mask = ~obs["is_last"]
            acts = {k: self._mask(v, mask) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()
        self.acts = acts
        trans = {**obs, **acts, **outs}
        for i in range(self.length):
            trn = {k: v[i] for k, v in trans.items()}
            [fn(trn, i, **self.kwargs) for fn in self.callbacks]
        step += len(obs["is_first"])
        episode += obs["is_last"].sum()
        return step, episode

    def _mask(self, value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)
