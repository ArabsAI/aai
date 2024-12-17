"""Whisper model building blocks."""

import torch
import torch.nn as nn

from aai.config import Config
from aai.modeling.architectures import register_architecture
from aai.modeling.modules.whisper_blocks import WhisperSpeechEncoder, WhisperTextDecoder


@register_architecture
class Whisper(nn.Module):
    """Whisper model for speech-to-text tasks."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = WhisperSpeechEncoder(config=self.config)
        self.decoder = WhisperTextDecoder(config=self.config)

    def forward(
        self, batch: dict[str, torch.Tensor], training: bool = True
    ) -> torch.Tensor:
        batch.get("audio", torch.ones((8, 32, 8)))
        text = batch.get("text", torch.ones((8, 32, 8)))

        encoder_output = self.encoder(batch, training)
        return self.decoder(text, encoder_output["x"], training)[0]
