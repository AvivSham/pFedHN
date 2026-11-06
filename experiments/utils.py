import argparse
import logging
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


# Re-export quantization utilities
from experiments.quantization import (
    calc_compression_stats,
    dequantize_delta_theta,
    dequantize_hypernetwork_weights,
    dequantize_tensor,
    quantize_delta_theta,
    quantize_hypernetwork_weights,
    quantize_tensor,
)

__all__ = [
    "calc_compression_stats",
    "count_parameters",
    "dequantize_delta_theta",
    "dequantize_hypernetwork_weights",
    "dequantize_tensor",
    "get_device",
    "quantize_delta_theta",
    "quantize_hypernetwork_weights",
    "quantize_tensor",
    "set_logger",
    "set_seed",
    "str2bool",
]
