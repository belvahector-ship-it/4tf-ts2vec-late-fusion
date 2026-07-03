"""
tests/test_device.py

Unit tests for src/utils/device.py (M0 — Project Bootstrap, ADR-007).

Covers IMP-01 M0 Definition of Done:
- "get_device() returns 'cuda' when CUDA is available and 'cpu'
  otherwise, without error on either."
"""

from __future__ import annotations

import torch

from src.utils.device import get_device, get_device_info


class TestGetDevice:
    def test_returns_torch_device(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)

    def test_never_raises(self) -> None:
        """Must not raise regardless of whether CUDA is available."""
        get_device()  # implicit assertion: no exception

    def test_matches_cuda_availability(self) -> None:
        device = get_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"


class TestGetDeviceInfo:
    def test_returns_dict_with_required_keys(self) -> None:
        info = get_device_info()
        assert "device" in info
        assert "cuda_available" in info
        assert isinstance(info["cuda_available"], bool)

    def test_cuda_specific_keys_present_only_if_available(self) -> None:
        info = get_device_info()
        if info["cuda_available"]:
            assert "device_name" in info
            assert "cuda_device_count" in info
        else:
            assert "device_name" not in info
