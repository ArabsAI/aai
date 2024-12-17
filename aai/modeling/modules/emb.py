import torch
import torch.nn as nn

from aai.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """Implements sinusoidal positional encoding."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_embeddings = self.config.arch.max_pos_emb_length
        self.embedding_dim = self.config.arch.embedding_dim

        position = torch.arange(0, self.num_embeddings).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / self.embedding_dim)
        )

        pe = torch.zeros(self.num_embeddings, self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[: x.size(1), :]
        return x + pe.unsqueeze(0)


class Embedding(nn.Module):
    """Token and positional embedding layer."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(
            num_embeddings=self.config.arch.vocab_size,
            embedding_dim=self.config.arch.embedding_dim,
        )

        self.positional_embedding_type = self.config.arch.positional_embedding_type
        if self.positional_embedding_type == "learned":
            self.wpe = nn.Embedding(
                num_embeddings=self.config.arch.max_pos_emb_length,
                embedding_dim=self.config.arch.embedding_dim,
            )
        elif self.positional_embedding_type == "sinusoidal":
            self.wpe = SinusoidalPositionalEncoding(config)
        else:
            raise ValueError(
                f"Unknown positional embedding type: {self.positional_embedding_type}"
            )

        self.dropout = nn.Dropout(self.config.arch.residual_dropout_rate)

    def forward(
        self, batch: dict[str, torch.Tensor], training: bool, attend: bool = False
    ) -> dict[str, torch.Tensor]:
        if attend:
            inputs = batch.pop("x")
            logits = torch.matmul(inputs, self.wte.weight.t())
            batch["logits"] = logits
        else:
            inputs = batch["inputs"].long()
            inputs = torch.clamp(inputs, max=self.config.arch.vocab_size - 1)

            embeddings = self.wte(inputs)

            if self.positional_embedding_type == "learned":
                positions = torch.arange(0, inputs.size(-1), device=inputs.device)
                position_embeddings = self.wpe(positions)
                embeddings = embeddings + position_embeddings
            else:  # sinusoidal
                embeddings = self.wpe(embeddings)

            embeddings = self.dropout(embeddings)
            batch["x"] = embeddings

        return batch
