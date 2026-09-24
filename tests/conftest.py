"""
tests/conftest.py
==================
Pytest configuration for NeuroForge test suite.
Sets deterministic random seeds for all frameworks.
"""
import numpy as np
import torch
import pytest

from neuroforge.core.config import SEED


@pytest.fixture(autouse=True)
def set_seeds():
    """Auto-use fixture: set deterministic seeds before each test."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    yield
