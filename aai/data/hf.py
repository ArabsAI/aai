from collections.abc import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from aai.config import Config
from aai.data.processors import BatchProcessor
from aai.data.tokenizers import get_tokenizer


class HuggingFaceDataset(IterableDataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._tokenizer = get_tokenizer(config)
        self._processor = BatchProcessor(config, tokenizer=self._tokenizer)
        self._dataset = load_dataset(
            path=self.config.data.path,
            name=self.config.data.name,
            split=self.config.data.split,
            streaming=self.config.data.streaming,
            trust_remote_code=True,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        chunk_size = self.config.data.batch_size * self.config.arch.max_sequence_length
        token_buffer = []
        attention_mask_buffer = []

        for example in self._dataset:
            tokens, attention_mask = self._processor(example)
            token_buffer.extend(tokens)
            attention_mask_buffer.extend(attention_mask)

            while len(token_buffer) > chunk_size + 1:
                batch = {
                    "inputs": torch.tensor(token_buffer[:chunk_size]).reshape(
                        self.config.data.batch_size, -1
                    ),
                    "targets": torch.tensor(token_buffer[1 : chunk_size + 1]).reshape(
                        self.config.data.batch_size, -1
                    ),
                    "mask": torch.tensor(
                        attention_mask_buffer[1 : chunk_size + 1]
                    ).reshape(self.config.data.batch_size, -1),
                }
                yield batch
                token_buffer = token_buffer[chunk_size:]
                attention_mask_buffer = attention_mask_buffer[chunk_size:]

    @property
    def sequence_length(self):
        return self.config.data.sequence_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)
