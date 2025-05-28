from .adapters import ModelAdapter, ModelWeights, TrainData, EvalData, Metrics
from .mock_model import MockModelAdapter

# This module provides a common interface for model adapters and includes a mock implementation for testing purposes.
__all__ = ['ModelAdapter', 'ModelWeights', 'TrainData', 'EvalData', 'Metrics', 'MockModelAdapter']