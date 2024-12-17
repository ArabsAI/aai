import torch
import torch.nn as nn

from aai.config import Config


class TransformerBlock(nn.Module):
    """Transformer block."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.head_dim = self.config.arch.embedding_dim // self.config.arch.n_heads

        self.layernorm_1 = nn.LayerNorm(self.config.arch.embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.arch.embedding_dim,
            num_heads=self.config.arch.n_heads,
            dropout=self.config.arch.residual_dropout_rate,
            batch_first=True,
        )

        self.layernorm_2 = nn.LayerNorm(self.config.arch.embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.config.arch.embedding_dim, self.config.arch.ffn_dim),
            nn.GELU(),
            nn.Dropout(self.config.arch.residual_dropout_rate),
            nn.Linear(self.config.arch.ffn_dim, self.config.arch.embedding_dim),
            nn.Dropout(self.config.arch.residual_dropout_rate),
        )

    def forward(
        self, batch: dict[str, torch.Tensor], training: bool
    ) -> dict[str, torch.Tensor]:
        x = batch.pop("x")
        attention_mask = batch.get("mask")

        # Self Attention
        residual = x
        x = self.layernorm_1(x)
        x, _ = self.attention(
            x, x, x, key_padding_mask=attention_mask, need_weights=False
        )
        x = residual + x

        # Feed Forward
        residual = x
        x = self.layernorm_2(x)
        x = self.ffn(x)
        x = residual + x

        batch["x"] = x
        return batch


class MTJTransformerBlock(nn.Module):
    """MTJ Transformer block."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.head_dim = self.config.arch.embedding_dim // self.config.arch.n_heads

        self.qkv_proj = nn.Linear(
            self.config.arch.embedding_dim,
            self.config.arch.n_heads
            * (
                3 * self.head_dim
                + (self.config.arch.ffn_dim // self.config.arch.n_heads)
            ),
            bias=False,
        )

        self.layernorm = nn.LayerNorm(self.config.arch.embedding_dim)

        self.output_proj = nn.Linear(
            self.config.arch.n_heads
            * (self.head_dim + (self.config.arch.ffn_dim // self.config.arch.n_heads)),
            self.config.arch.embedding_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(self.config.arch.residual_dropout_rate)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        training: bool,
    ) -> dict[str, torch.Tensor]:
        inputs = batch.pop("x")
        attn_mask = batch.get("mask")

        ln_inputs = self.layernorm(inputs)

        input_projected = self.qkv_proj(ln_inputs)
        q, k, v, ffn_out = torch.split(
            input_projected,
            [self.head_dim, self.head_dim * 2, self.head_dim * 3],
            dim=-1,
        )

        # Self Attention
        attention_fn = nn.MultiheadAttention(
            embed_dim=self.config.arch.embedding_dim,
            num_heads=self.config.arch.n_heads,
            dropout=self.config.arch.residual_dropout_rate,
            batch_first=True,
        )
        attn_out = attention_fn(
            query=q,
            value=v,
            key=k,
            mask=attn_mask,
        )  # [batch, seq, n_heads, head_dim]

        # Fused attn out and ffn red projection
        ffn_out = torch.nn.functional.gelu(ffn_out)
        fused_input = torch.cat((attn_out, ffn_out), dim=-1)

        output = self.output_proj(fused_input)

        inputs += self.dropout(output, training=training)

        batch.update({"x": inputs})
        return batch


class BlockwiseParallelTransformer(nn.Module):
    pass
