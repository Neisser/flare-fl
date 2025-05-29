from .adapters import EvalData, Metrics, ModelAdapter, ModelWeights, TrainData
from .mock_model import MockModelAdapter
from .pytorch_adapter import PyTorchModelAdapter

# This module provides a common interface for model adapters and includes a mock implementation for testing purposes.
__all__ = [
    "ModelAdapter",
    "ModelWeights",
    "TrainData",
    "EvalData",
    "Metrics",
    "MockModelAdapter",
    "PyTorchModelAdapter",
]
