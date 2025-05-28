from typing import Any, Dict
from .adapters import ModelAdapter, ModelInstance, ModelWeights, TrainData, EvalData, Metrics

class MockModelAdapter(ModelAdapter):
    """
    A mock model adapter for testing and simulation purposes.
    Simulates training and weight management without a real ML model.
    """
    def __init__(self, model_instance: ModelInstance = "MockModel"):
        super().__init__(model_instance)
        # Simulate weights as a simple dictionary
        self._weights: ModelWeights = {"param1": 0.5, "param2": 1.5}
        print(f"MockModelAdapter initialized for model: {model_instance}")

    def get_weights(self) -> ModelWeights:
        print("MockModelAdapter: get_weights called.")
        return self._weights.copy() # Returns a copy of the weights to avoid external modifications.

    def set_weights(self, weights: ModelWeights) -> None:
        print(f"MockModelAdapter: set_weights called with {weights}.")
        if not isinstance(weights, dict):
            raise ValueError("MockModelAdapter expects weights to be a dictionary.")
        self._weights = weights.copy()

    def train(self, data: TrainData, epochs: int, learning_rate: float, **kwargs) -> Dict[str, Any]:
        print(f"MockModelAdapter: train called for {epochs} epochs with lr={learning_rate} on data: {str(data)[:50]}...")
        # Simulate a simple improvement in loss based on the current weights.
        simulated_loss = self._weights.get("param1", 1.0) * 0.8 # Reducir loss
        self._weights["param1"] = simulated_loss # "learn" some new value
        print(f"MockModelAdapter: training finished, simulated new param1: {self._weights['param1']}")
        return {"loss": [simulated_loss + 0.1, simulated_loss], "epochs": epochs} # Simple history

    def evaluate(self, data: EvalData, **kwargs) -> Metrics:
        print(f"MockModelAdapter: evaluate called on data: {str(data)[:50]}...")
        # Simulate metrics based on the weights
        accuracy = min(1.0, 1.0 - self._weights.get("param1", 1.0) * 0.1)
        loss = self._weights.get("param1", 1.0) * 0.2
        print(f"MockModelAdapter: evaluation complete. Accuracy: {accuracy}, Loss: {loss}")
        return {"accuracy": accuracy, "loss": loss}

    def predict(self, data: Any, **kwargs) -> Any:
        print(f"MockModelAdapter: predict called on data: {str(data)[:50]}...")
        # Simulate a simple prediction based on the weights
        return [self._weights.get("param1", 0.5) * x for x in range(min(5, len(str(data))))]

    def serialize_model(self) -> bytes:
        print("MockModelAdapter: serialize_model called.")
        # Simulate serialization (e.g., using JSON or pickle for this mock)
        import json
        return json.dumps({"model_type": str(self.model), "weights": self._weights}).encode('utf-8')

    def deserialize_model(self, model_bytes: bytes) -> None:
        print("MockModelAdapter: deserialize_model called.")
        import json
        data = json.loads(model_bytes.decode('utf-8'))
        self.model = data.get("model_type", "DeserializedMockModel")
        self._weights = data.get("weights", {"param1": 0.0})
        print(f"MockModelAdapter: model deserialized to type {self.model} with weights {self._weights}")

    def serialize_weights(self) -> bytes:
        print("MockModelAdapter: serialize_weights called.")
        import json
        return json.dumps(self._weights).encode('utf-8')

    def deserialize_weights(self, weights_bytes: bytes) -> ModelWeights:
        print("MockModelAdapter: deserialize_weights called.")
        import json
        return json.loads(weights_bytes.decode('utf-8'))
        