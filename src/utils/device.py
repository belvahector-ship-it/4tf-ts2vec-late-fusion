"""
src/utils/device.py

Device detection utility (M0 — Project Bootstrap, ADR-007).

Purpose
-------
Returns the best available compute device without ever raising an
error, whether or not a GPU is present. Per IMP-01 M0 Definition of
Done: "get_device() returns 'cuda' when CUDA is available and 'cpu'
otherwise, without error on either."
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Return the best available torch device.

    Returns
    -------
    torch.device
        `torch.device("cuda")` if a CUDA-capable GPU is available,
        otherwise `torch.device("cpu")`. Never raises.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict[str, object]:
    """
    Return diagnostic information about the selected device.

    Useful for logging at the start of a training run so checkpoints
    and logs record what hardware actually produced a given result
    (relevant to IMP-01 Risk R-06, non-determinism between GPU runs).

    Returns
    -------
    dict
        Keys: "device" (str), "cuda_available" (bool), and, if CUDA is
        available, "device_name" (str) and "cuda_device_count" (int).
    """
    device = get_device()
    info: dict[str, object] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info
