from abc import ABC, abstractmethod
from typing import Any, Dict

# Placeholder types for model data, weights, etc.
ModelWeights = Any  # May also be a list of numpy arrays or pyTorch/TF tensors, etc.
ModelInstance = Any  # Real instance of the model (e.g., a PyTorch nn.Module object).
TrainData = Any
EvalData = Any
Metrics = Dict[str, float]


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    Provides a common interface for interacting with different ML frameworks.
    """

    def __init__(self, model_instance: ModelInstance):
        self.model = model_instance

    @abstractmethod
    def get_weights(self) -> ModelWeights:
        """Returns the current weights of the model."""
        pass

    @abstractmethod
    def set_weights(self, weights: ModelWeights) -> None:
        """Sets the weights of the model."""
        pass

    @abstractmethod
    def train(
        self, data: TrainData, epochs: int, learning_rate: float, **kwargs
    ) -> Dict[str, Any]:
        """
        Trains the model on the given data.
        Returns a dictionary with training history (e.g., loss).
        """
        pass

    @abstractmethod
    def evaluate(self, data: EvalData, **kwargs) -> Metrics:
        """
        Evaluates the model on the given data.
        Returns a dictionary of metrics (e.g., {'accuracy': 0.95, 'loss': 0.12}).
        """
        pass

    @abstractmethod
    def predict(self, data: Any, **kwargs) -> Any:
        """Makes predictions on the given data."""
        pass

    @abstractmethod
    def serialize_model(self) -> bytes:
        """Serializes the entire model (architecture + weights) to bytes."""
        pass

    @abstractmethod
    def deserialize_model(self, model_bytes: bytes) -> None:
        """Deserializes and loads the model from bytes."""
        pass

    @abstractmethod
    def serialize_weights(self) -> bytes:
        """Serializes only the model weights to bytes."""
        pass

    @abstractmethod
    def deserialize_weights(self, weights_bytes: bytes) -> ModelWeights:
        """Deserializes model weights from bytes and returns them (does not set them)."""
        pass
