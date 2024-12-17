"""Transformer model building blocks."""

import torch
import torch.nn as nn

from aai.config import Config
from aai.modeling.architectures import register_architecture
from aai.modeling.modules.emb import Embedding
from aai.modeling.modules.transformer_block import TransformerBlock


@register_architecture
class Transformer(nn.Module):
    """Transformer model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = Embedding(self.config)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.arch.n_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(self.config.arch.embedding_dim)

    def forward(
        self, batch: dict[str, torch.Tensor], training: bool = True
    ) -> dict[str, torch.Tensor]:
        batch = self.embedding(batch, training)

        for block in self.transformer_blocks:
            batch = block(batch, training)

        x = batch.pop("x")
        x = self.final_layer_norm(x)
        batch["x"] = x

        batch = self.embedding(batch, training, attend=True)
        return batch
