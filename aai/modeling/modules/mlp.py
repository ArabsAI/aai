import torch
import torch.nn as nn

from aai.config import Config
from aai.modeling import register_model


@register_model
class MLP(nn.Module):
    config: Config

    def __init__(self, config: Config):
        super(MLP, self).__init__()
        self.config = config

        self.w1 = nn.Linear(
            in_features=config.architecture.ffn_dim,
            out_features=config.architecture.ffn_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            in_features=config.architecture.ffn_dim,
            out_features=config.architecture.embedding_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            in_features=config.architecture.ffn_dim,
            out_features=config.architecture.ffn_dim,
            bias=False,
        )
        self.dropout = nn.Dropout(p=config.architecture.residual_dropout)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        x = self.w2(torch.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, training=not deterministic)
        return x
