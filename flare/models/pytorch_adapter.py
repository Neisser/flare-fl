import pickle
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from .adapters import EvalData, Metrics, ModelAdapter, ModelWeights, TrainData


class PyTorchModelAdapter(ModelAdapter):
    """
    ModelAdapter implementation for PyTorch models.

    Handles PyTorch nn.Module objects and provides serialization/deserialization
    functionality compatible with the Flare ecosystem.
    """

    def __init__(self, model_instance: nn.Module):
        """
        Initialize the PyTorch model adapter.

        Args:
            model_instance: A PyTorch nn.Module instance
        """
        super().__init__(model_instance)
        self.device = (
            next(model_instance.parameters()).device
            if list(model_instance.parameters())
            else torch.device("cpu")
        )
        print(f"PyTorchModelAdapter initialized with model on device: {self.device}")

    def get_weights(self) -> ModelWeights:
        """
        Returns the current weights of the PyTorch model as a dictionary of tensors.
        """
        # Return state_dict as ModelWeights (dictionary of parameter tensors)
        return {
            name: param.data.clone() for name, param in self.model.state_dict().items()
        }

    def set_weights(self, weights: ModelWeights) -> None:
        """
        Sets the weights of the PyTorch model from a dictionary of tensors.
        """
        if isinstance(weights, dict):
            # Create a state dict and load it
            state_dict = {}
            for name, weight in weights.items():
                if hasattr(weight, "to"):
                    # Already a PyTorch tensor
                    state_dict[name] = weight.to(self.device)
                elif hasattr(weight, "shape"):
                    # Numpy array - convert to PyTorch tensor
                    tensor = torch.from_numpy(weight).to(self.device)
                    state_dict[name] = tensor
                else:
                    # Fallback for other types
                    state_dict[name] = weight

            self.model.load_state_dict(state_dict)
        else:
            raise ValueError(
                f"Expected weights to be a dictionary, got {type(weights)}"
            )

    def train(
        self, data: TrainData, epochs: int, learning_rate: float, **kwargs
    ) -> Dict[str, Any]:
        """
        Trains the PyTorch model on the given data.

        Args:
            data: Tuple of (features, labels) or DataLoader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training history (losses)
        """
        self.model.train()
        # Setup optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)  # type: ignore
        criterion = nn.CrossEntropyLoss()

        history = {"losses": []}

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            if isinstance(data, tuple) and len(data) == 2:
                # Simple (X, y) format
                X, y = data
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss = loss.item()
                num_batches = 1

            else:
                # Assume it's a DataLoader
                try:
                    for batch_X, batch_y in data:
                        batch_X, batch_y = (
                            batch_X.to(self.device),
                            batch_y.to(self.device),
                        )

                        optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        num_batches += 1
                except Exception as e:
                    print(f"Error processing data loader: {e}")
                    raise

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            history["losses"].append(avg_loss)

            if epoch % max(1, epochs // 5) == 0:  # Log every 20% of epochs
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return history

    def evaluate(self, data: EvalData, **kwargs) -> Metrics:
        """
        Evaluates the PyTorch model on the given data.

        Args:
            data: Tuple of (features, labels) or DataLoader
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of metrics (accuracy, loss)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            if isinstance(data, tuple) and len(data) == 2:
                # Simple (X, y) format
                X, y = data
                X, y = X.to(self.device), y.to(self.device)

                outputs = self.model(X)
                loss = criterion(outputs, y)
                _, predicted = torch.max(outputs.data, 1)

                total_loss = loss.item()
                total_correct = (predicted == y).sum().item()
                total_samples = y.size(0)

            else:
                # Assume it's a DataLoader
                for batch_X, batch_y in data:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs.data, 1)

                    total_loss += loss.item()
                    total_correct += (predicted == batch_y).sum().item()
                    total_samples += batch_y.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / max(1, total_samples)

        return {"accuracy": accuracy, "loss": avg_loss, "total_samples": total_samples}

    def predict(self, data: Any, **kwargs) -> Any:
        """
        Makes predictions on the given data.

        Args:
            data: Input features tensor or DataLoader
            **kwargs: Additional prediction parameters

        Returns:
            Predicted class labels or probabilities
        """
        self.model.eval()

        with torch.no_grad():
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                return predicted
            else:
                # Assume it's a DataLoader
                predictions = []
                for batch_X in data:
                    if isinstance(batch_X, (tuple, list)):
                        batch_X = batch_X[
                            0
                        ]  # Extract features if it's (features, labels)
                    batch_X = batch_X.to(self.device)
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.append(predicted)
                return torch.cat(predictions)

    def serialize_model(self) -> bytes:
        """
        Serializes the entire PyTorch model (architecture + weights) to bytes.
        """
        # Save the entire model using pickle
        # In production, consider using torch.save for better compatibility
        return pickle.dumps(self.model)

    def deserialize_model(self, model_bytes: bytes) -> None:
        """
        Deserializes and loads the PyTorch model from bytes.
        """
        self.model = pickle.loads(model_bytes)
        self.model = self.model.to(self.device)

    def serialize_weights(self) -> bytes:
        """
        Serializes only the PyTorch model weights (state_dict) to bytes.
        """
        # Use pickle to serialize the state_dict
        state_dict = self.model.state_dict()
        # Convert tensors to CPU to avoid device-specific serialization issues
        cpu_state_dict = {name: param.cpu() for name, param in state_dict.items()}
        return pickle.dumps(cpu_state_dict)

    def deserialize_weights(self, weights_bytes: bytes) -> ModelWeights:
        """
        Deserializes PyTorch model weights from bytes and returns them.
        Does not set them on the model.
        """
        state_dict = pickle.loads(weights_bytes)
        # Convert back to appropriate device
        return {name: param.to(self.device) for name, param in state_dict.items()}
        return {name: param.to(self.device) for name, param in state_dict.items()}
