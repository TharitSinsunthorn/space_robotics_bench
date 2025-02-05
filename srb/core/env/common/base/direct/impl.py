from functools import cached_property
from typing import Dict, Sequence, Tuple

import gymnasium
import numpy
import torch
from isaaclab.envs import DirectRLEnv as __DirectRLEnv

from srb._typing import IntermediateTaskState
from srb.core.manager import ActionManager

from .cfg import DirectEnvCfg


class __PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class DirectEnv(__DirectRLEnv, metaclass=__PostInitCaller):
    cfg: DirectEnvCfg
    _internal_state: IntermediateTaskState

    def __init__(self, cfg: DirectEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Apply visuals
        self.cfg.visuals.func(self.cfg.visuals)

        # Introduce action manager
        if self.cfg.actions:
            self.action_manager = ActionManager(
                self.cfg.actions,
                env=self,  # type: ignore
            )
            print("[INFO] Action Manager: ", self.action_manager)

    def close(self):
        if not self._is_closed:
            if self.cfg.actions:
                del self.action_manager

        super().close()

    def _reset_idx(self, env_ids: Sequence[int]):
        if self.cfg.actions:
            self.action_manager.reset(env_ids)

        if self.cfg.events:
            self.event_manager.reset(env_ids)

        super()._reset_idx(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.cfg.actions:
            self.action_manager.process_action(actions)
        else:
            super()._pre_physics_step(actions)  # type: ignore

    def _apply_action(self):
        if self.cfg.actions:
            self.action_manager.apply_action()
        else:
            super()._apply_action()  # type: ignore

    def __post_init__(self):
        if self._implements_internal_state_workflow:
            # Initialize the intermediate state
            self._update_internal_state()

            ## Verify that _internal_state is correctly initialized
            assert hasattr(self, "_internal_state"), (
                "Internal state should be initialized in the constructor."
            )
            assert isinstance(self._internal_state, Dict), (
                f"Internal state should be a dictionary. "
                f"Actual: {type(self._internal_state)}"
            )
            assert "obs" in self._internal_state, (
                "Internal state should have an 'obs' key."
            )
            assert "rew" in self._internal_state, (
                "Internal state should have a 'rew' key."
            )
            assert "term" in self._internal_state, (
                "Internal state should have a 'term' key."
            )
            assert "trunc" in self._internal_state, (
                "Internal state should have a 'trunc' key."
            )

            ## Verify that all reward components have the correct shape
            for rew_key, rew_val in self._internal_state["rew"].items():
                assert rew_val.shape == (self.num_envs,), (
                    f"Reward component '{rew_key}' has an incorrect shape. "
                    f"Expected: ({self.num_envs},) | Actual: {rew_val.shape}"
                )

            ## Verify that all observation components have the correct shape
            for obs_key, obs_val in self._internal_state["obs"].items():
                for obs_sub_key, obs_sub_val in obs_val.items():
                    assert obs_sub_val.size(0) == self.num_envs, (
                        f"Observation component '{obs_key}/{obs_sub_key}' has an incorrect shape. "
                        f"Expected: ({self.num_envs}, ...) | Actual: {obs_sub_val.shape}"
                    )

        # Automatically determine the action and observation spaces for all sub-classes
        self._update_gym_env_spaces()

    def _update_gym_env_spaces(self):
        # Action space
        self.single_action_space = gymnasium.spaces.Box(
            low=-numpy.inf,
            high=numpy.inf,
            shape=(self.action_manager.total_action_dim,),
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

        # Observation space
        self.single_observation_space = gymnasium.spaces.Dict({})
        for obs_key, obs_buf in self._get_observations().items():
            assert isinstance(obs_buf, (numpy.ndarray, torch.Tensor))
            self.single_observation_space[obs_key] = gymnasium.spaces.Box(
                low=-numpy.inf, high=numpy.inf, shape=obs_buf.shape[1:]
            )
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    @cached_property
    def max_episode_length(self):
        # Wrap lengthy calculation in a cached property
        return super().max_episode_length

    def _update_internal_state(self):
        raise NotImplementedError
        self._internal_state = ...

    @cached_property
    def _implements_internal_state_workflow(self) -> bool:
        # Check if the class implements the intermediate state
        return (
            hasattr(self.__class__, "_update_internal_state")
            and self.__class__._update_internal_state
            is not DirectEnv._update_internal_state
        )

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._implements_internal_state_workflow:
            self._update_internal_state()
            if self.cfg.include_extras:
                self.extras.update(
                    {
                        "obs": self._internal_state["obs"],
                        "rew": self._internal_state["rew"],
                    }
                )
            return self._internal_state["term"], self._internal_state["trunc"]
        else:
            return super()._get_dones()  # type: ignore

    def _get_rewards(self) -> torch.Tensor:
        if self._implements_internal_state_workflow:
            return _sum_rewards(self._internal_state["rew"])
        else:
            return super()._get_rewards()  # type: ignore

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        if self._implements_internal_state_workflow:
            return _flatten_observations(self._internal_state["obs"])
        else:
            return super()._get_observations()  # type: ignore


@torch.jit.script
def _flatten_observations(
    obs_dict: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    return {
        obs_cat: torch.cat(
            [
                torch.flatten(obs_group[obs_key], start_dim=1)
                for obs_key in sorted(obs_group.keys())
            ],
            dim=1,
        )
        for obs_cat, obs_group in obs_dict.items()
    }


@torch.jit.script
def _sum_rewards(rew_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sum(torch.stack(list(rew_dict.values()), dim=1), dim=1)
