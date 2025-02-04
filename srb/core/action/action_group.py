from __future__ import annotations

from typing import ClassVar, Iterable, List, Sequence, Type

import torch

from srb.utils.cfg import configclass
from srb.utils.str import convert_to_snake_case


@configclass
class ActionGroup:
    def map_commands(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert convert_to_snake_case(cls.__name__) not in (
            convert_to_snake_case(action_group.__name__)
            for action_group in ActionGroupRegistry.registry
        ), (
            f"Cannot register multiple action groups with an identical name: '{cls.__module__}:{cls.__name__}' already exists as '{next(robot for robot in ActionGroupRegistry.registry if convert_to_snake_case(cls.__name__) == convert_to_snake_case(robot.__name__)).__module__}:{cls.__name__}'"
        )
        ActionGroupRegistry.registry.append(cls)

    @classmethod
    def action_group_registry(cls) -> Sequence[Type[ActionGroup]]:
        return ActionGroupRegistry.registry


class ActionGroupRegistry:
    registry: ClassVar[List[Type[ActionGroup]]] = []

    @classmethod
    def __len__(cls) -> int:
        return len(cls.registry)

    @classmethod
    def registered_modules(cls) -> Iterable[str]:
        return {action_group.__module__ for action_group in cls.registry}

    @classmethod
    def registered_packages(cls) -> Iterable[str]:
        return {module.split(".", maxsplit=1)[0] for module in cls.registered_modules()}

    @classmethod
    def get_by_name(cls, name: str) -> Type[ActionGroup] | None:
        for action_group in cls.registry:
            if convert_to_snake_case(action_group.__name__) == name:
                return action_group
        return None
