import logging

import torch
from rich.logging import RichHandler

from aai.config import Config
from aai.core.cortex import Cortex
from aai.data.hf import HuggingFaceDataset
from aai.utils.parser import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=True,
            tracebacks_show_locals=True,
        )
    ],
)


def main() -> None:
    logging.info(
        f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    args = parse_args()
    config = Config.read_config_from_yaml(args.config_path)
    logging.info(f"Config: {config}")

    dataset = HuggingFaceDataset(config)
    cortex = Cortex(config)
    cortex.train(dataset)


if __name__ == "__main__":
    main()
