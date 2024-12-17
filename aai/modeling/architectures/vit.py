"""Vit model building blocks."""

import flax.linen as nn
import jax

from aai.config import Config
from aai.modeling.architectures import register_architecture
from aai.modeling.modules.emb import PatchEmbedding
from aai.modeling.modules.vit_block import ViTBlock


@register_architecture
class ViT(nn.Module):
    """Implements the encoder component of the Vision Transformer (ViT)
    model.
    """

    config: Config

    def setup(self):
        self.embedding = PatchEmbedding(config=self.config, name="patch_embedding")

        self.vit_blocks: list[ViTBlock] = [
            ViTBlock(self.config, name=f"vit_block_{idx}")
            for idx in range(self.config.arch.n_layers)
        ]
        self.dropout = nn.Dropout(
            self.config.arch.residual_dropout_rate, name="dropout"
        )
        self.output_layer = nn.Dense(self.config.arch.n_outputs, name="output_layer")

    def __call__(
        self,
        batch: dict[str, jax.Array],
        training: bool = False,
    ) -> dict[str, jax.Array]:
        batch = self.embedding(batch, training)
        for idx in range(self.config.arch.n_layers):
            batch = self.vit_blocks[idx](batch, training)
        x = batch.pop("x")
        x = self.dropout(x, deterministic=not training)
        x = self.output_layer(x[:, 0, :])
        batch.update({"x": x})
        return batch
