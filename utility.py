"""
Normalization utilities for GLONET ocean forecasting model.

This module provides functions to load and apply normalization statistics
(mean and standard deviation) for different depth levels of oceanographic variables.
These statistics are used to normalize input data before model inference and
denormalize predictions after inference.

"""
from torchvision import transforms
import numpy as np

def _get_transform(model_dir: str, level_group: int, denormalize: bool = False):
    """
    Generic function to create a normalizer or denormalizer transform.

    Args:
        model_dir: Path to the model directory containing statistics files.
        level_group: Integer indicating the depth group (1, 2, or 3).
        denormalize: If True, creates a denormalizer; otherwise, a normalizer.

    Returns:
        A torchvision.transforms.Normalize object.
    """
    if level_group == 1:
        levels = ["L0"]
        variables = ["zos", "thetao", "so", "uo", "vo"]
    elif level_group == 2:
        levels = [
            "L50", "L100", "L150", "L222", "L318", "L380", "L450", "L540", "L640", "L763"
        ]
        variables = ["thetao", "so", "uo", "vo"]
    elif level_group == 3:
        levels = [
            "L902", "L1245", "L1684", "L2225", "L3220", "L3597", "L3992", "L4405", "L4833", "L5274"
        ]
        variables = ["thetao", "so", "uo", "vo"]
    else:
        raise ValueError("level_group must be 1, 2, or 3")

    # Load stats and build mean and std arrays
    mean_parts = []
    std_parts = []
    for var in variables:
        for level in levels:
            mean_path = f"{model_dir}/{level}/{var}_mean.npy"
            std_path = f"{model_dir}/{level}/{var}_std.npy"
            mean_parts.append(np.load(mean_path))
            std_parts.append(np.load(std_path))

    gmean = np.concatenate(mean_parts)
    gstd = np.concatenate(std_parts)

    if denormalize:
        # The formula to reverse a normalization is:
        # new_mean = -mean / std
        # new_std = 1 / std
        mean_for_transform = [-m / s for m, s in zip(gmean, gstd)]
        std_for_transform = [1 / s for s in gstd]
    else:
        mean_for_transform = gmean
        std_for_transform = gstd

    return transforms.Normalize(mean=mean_for_transform, std=std_for_transform)


# LEVEL 1 (Surface): L0
def get_normalizer1(model_dir: str):
    return _get_transform(model_dir, 1)

def get_denormalizer1(model_dir: str):
    return _get_transform(model_dir, 1, denormalize=True)

# LEVEL 2 (Intermediate depths):
def get_normalizer2(model_dir: str):
    return _get_transform(model_dir, 2)

def get_denormalizer2(model_dir: str):
    return _get_transform(model_dir, 2, denormalize=True)

# LEVEL 3 (Deep ocean):
def get_normalizer3(model_dir: str):
    return _get_transform(model_dir, 3)

def get_denormalizer3(model_dir: str):
    return _get_transform(model_dir, 3, denormalize=True)
