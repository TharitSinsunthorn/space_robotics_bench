from functools import cached_property
from typing import Any, Mapping, Sequence, Tuple

import gymnasium
import torch
from skrl.envs.wrappers.torch import IsaacLabWrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class SkrlEnvWrapper(IsaacLabWrapper):
    def __init__(
        self,
        env: Any,
        obs_keys: Sequence[str] = [],
        state_keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(env)
        self._obs_keys = obs_keys
        self._state_keys = state_keys

    @cached_property
    def observation_space(self) -> gymnasium.Space:
        if hasattr(self._unwrapped, "single_observation_space"):
            obs_space = self._unwrapped.single_observation_space
        else:
            obs_space = self._unwrapped.observation_space

        if self._obs_keys:
            return gymnasium.spaces.Dict(
                {key: obs_space[key] for key in self._obs_keys}
            )
        else:
            return obs_space

    @cached_property
    def state_space(self) -> gymnasium.Space | None:
        """State space"""
        if hasattr(self._unwrapped, "state_space"):
            return self._unwrapped.state_space

        if hasattr(self._unwrapped, "single_observation_space"):
            obs_space = self._unwrapped.single_observation_space
        else:
            obs_space = self._unwrapped.observation_space

        if self._state_keys is None:
            return None
        elif self._state_keys:
            return gymnasium.spaces.Dict(
                {key: obs_space[key] for key in self._state_keys}
            )
        else:
            return obs_space

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, reward, terminated, truncated, self._info = self._env.step(
            actions
        )
        self._observations = flatten_tensorized_space(
            tensorize_space(
                self.observation_space, self.__extract_observations(observations)
            )
        )
        return (
            self._observations,
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            self._info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = flatten_tensorized_space(
                tensorize_space(
                    self.observation_space, self.__extract_observations(observations)
                )
            )
            self._reset_once = False
        return self._observations, self._info

    def __extract_observations(
        self, observations: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor] | torch.Tensor:
        if not self._obs_keys:
            return observations
        return {key: observations[key] for key in self._obs_keys}
