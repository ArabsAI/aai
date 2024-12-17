import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from aai.config import Config


def make_optimizer(config: Config, model: torch.nn.Module):
    """Create optimizer with learning rate scheduler."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.optim.learning_rate,
        betas=(config.optim.beta1, config.optim.beta2),
    )

    if config.optim.lr_decay:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.optim.total_steps, eta_min=config.optim.lr_min
        )
        return optimizer, scheduler

    return optimizer, None
