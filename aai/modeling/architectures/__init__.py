from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.nn as nn

from aai.config import Config

_ARCHITECTURES: dict[str, Any] = {}  # registry


def register_architecture(cls):
    _ARCHITECTURES[cls.__name__.lower()] = cls
    return cls


class Architecture(ABC, nn.Module):
    """Base class for all architectures."""

    config: Config

    @abstractmethod
    def forward(
        self, batch: dict[str, torch.Tensor], training: bool
    ) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def shard(
        self, ps: torch.distributed.ProcessGroup
    ) -> Tuple[Architecture, torch.distributed.ProcessGroup]:
        pass


from aai.modeling.architectures.transformer import *  # isort:skip

# from aai.modeling.architectures.clip import *  # isort:skip
# from aai.modeling.architectures.mamba import *  # isort:skip
# from aai.modeling.architectures.vit import *  # isort:skip
# from aai.modeling.architectures.whisper import *  # isort:skip


def get_architecture(config: Config) -> Architecture:
    assert config.arch.architecture_name, "Arch config must specify 'architecture'."
    return _ARCHITECTURES[config.arch.architecture_name.lower()](config)
