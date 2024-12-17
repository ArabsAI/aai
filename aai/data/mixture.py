from collections.abc import Mapping
from typing import TypeVar, Union

import torch
import torch.distributions as dist
from torch.utils.data import Dataset

T = TypeVar("T")


class MixtureDataset(Dataset):
    """MixtureDataset supports loading data from multiple datasets.

    It takes a list of datasets and yields from them according to the
    weights.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        weights: dict[str, float],
        key: Union[int, torch.Tensor] = 0,
    ) -> None:
        self.datasets = datasets
        self.weights = MixtureDataset._normalize_weights(weights)

        if not isinstance(key, int):
            key = torch.randint(0, 2**20, (1,))

        self.key = key

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        """Normalize the weights to sum to 1."""
        total: float = sum(weights.values())
        if total == 0:
            raise ValueError(f"Datasets' weights cannot sum to 0, got {weights}")
        return {name: weight / total for name, weight in weights.items() if weight > 0}

    def shard(self, shard_id: int, num_shards: int) -> "MixtureDataset":
        """Return a MixtureDataset with the sharded datasets."""
        sharded = {
            name: dset.shard(shard_id, num_shards)
            for name, dset in self.datasets.items()
        }
        return MixtureDataset(sharded, self.weights)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, index: int) -> T:
        rng = dist.Categorical(torch.tensor(list(self.weights.values())))
        dataset_name = list(self.weights.keys())[rng.sample()]
        return self.datasets[dataset_name][index]
