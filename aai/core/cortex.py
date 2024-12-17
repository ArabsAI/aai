import torch
from accelerate import Accelerator
from tqdm import tqdm

from aai.config import Config
from aai.metrics.cross_entropy import CrossEntropyLoss
from aai.modeling.architectures import get_architecture
from aai.modeling.optimizers import make_optimizer


class Cortex:
    """The Cortex class represents the core component of the neural network model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.architecture = get_architecture(self.config)
        self.initialize_train_state()

    def initialize_train_state(self) -> None:
        """Initializes the train state of the model."""
        self.model = self.architecture.to(self.device)
        self.optimizer, self.scheduler = make_optimizer(
            config=self.config, model=self.model
        )

        # Move to accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

    def train(self, dataset) -> None:
        """Trains the model using the given dataset."""
        self.model.train()
        criterion = CrossEntropyLoss()

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        dataloader = self.accelerator.prepare(dataloader)

        pbar = tqdm(range(self.config.aai.total_steps))
        losses = []

        for step in pbar:
            batch = next(iter(dataloader))
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = criterion(outputs["logits"], batch["targets"])

            self.accelerator.backward(loss)
            self.optimizer.step()

            losses.append(loss.item())

            if len(losses) % self.config.aai.log_interval == 0:
                avg_loss = (
                    sum(losses[-self.config.aai.log_interval :])
                    / self.config.aai.log_interval
                )
                pbar.set_description(f"Step: {step} | Loss: {avg_loss:.4f}")
