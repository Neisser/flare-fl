from typing import Optional

from flare.core import RoundContext
from flare.federation.client import Client
from flare.models import ModelWeights


class FederatedClient(Client):
    """
    Enhanced client for federated learning that computes weight differences (ΔW)
    and applies compression before sending updates to the orchestrator.

    This client focuses on efficient transmission of model updates by:
    1. Computing the difference between local and global weights
    2. Compressing the difference using the configured compressor
    3. Only transmitting the compressed differences
    """

    def __init__(self, client_id: str, local_data, config):
        super().__init__(client_id, local_data, config)
        self.initial_global_weights: Optional[ModelWeights] = None
        print(
            f"FederatedClient {self.node_id} initialized with compression-aware training."
        )

    def receive_global_model(self, model_ref) -> bool:
        """
        Receives the global model and stores the initial weights for difference computation.
        """
        success = super().receive_global_model(model_ref)
        if success and self.current_global_model_weights is not None:
            # Store the initial global weights to compute ΔW later
            self.initial_global_weights = self._copy_weights(
                self.current_global_model_weights
            )
            print(
                f"FederatedClient {self.node_id}: Initial global weights stored for ΔW computation."
            )
        return success

    def train_local(
        self, round_context: RoundContext, epochs: int, learning_rate: float
    ) -> Optional[ModelWeights]:
        """
        Performs local training and returns the weight difference (ΔW) instead of full weights.
        """
        if self.current_global_model_weights is None:
            print(
                f"FederatedClient {self.node_id}: Cannot train, no global model received yet."
            )
            return None

        if self.initial_global_weights is None:
            print(
                f"FederatedClient {self.node_id}: Cannot compute ΔW, initial weights not stored."
            )
            return None

        print(
            f"FederatedClient {self.node_id}: Starting local training for round {round_context.round_number}..."
        )

        try:
            # Set the global model weights to start local training from
            self.model_adapter.set_weights(self.current_global_model_weights)

            # Perform local training
            train_history = self.model_adapter.train(
                data=self.local_data, epochs=epochs, learning_rate=learning_rate
            )
            print(
                f"FederatedClient {self.node_id}: Local training complete. History: {train_history}"
            )

            # Get the updated weights after training
            updated_weights = self.model_adapter.get_weights()

            # Compute weight difference (ΔW = W_local - W_global)
            delta_weights = self._compute_weight_difference(
                updated_weights, self.initial_global_weights
            )
            print(f"FederatedClient {self.node_id}: Computed weight difference (ΔW).")

            return delta_weights

        except Exception as e:
            print(f"FederatedClient {self.node_id}: Error during local training: {e}")
            return None

    def send_update(
        self, round_context: RoundContext, delta_weights: ModelWeights
    ) -> Optional[str]:
        """
        Compresses the weight differences (ΔW) and sends them to the storage provider.
        """
        if delta_weights is None:
            print(f"FederatedClient {self.node_id}: No weight differences to send.")
            return None

        print(
            f"FederatedClient {self.node_id}: Serializing and compressing weight differences..."
        )

        try:
            # Serialize the weight differences to bytes
            # Note: We need to temporarily set the model weights to delta_weights to serialize them
            original_weights = self.model_adapter.get_weights()
            self.model_adapter.set_weights(delta_weights)
            delta_bytes = self.model_adapter.serialize_weights()
            self.model_adapter.set_weights(original_weights)  # Restore original weights

            # Compress the serialized weight differences
            compressed_delta_bytes = self.compressor.compress(delta_bytes)
            compression_ratio = (
                len(delta_bytes) / len(compressed_delta_bytes)
                if len(compressed_delta_bytes) > 0
                else 1
            )
            print(
                f"FederatedClient {self.node_id}: ΔW compressed from {len(delta_bytes)} to {len(compressed_delta_bytes)} bytes (ratio: {compression_ratio:.2f}x)."
            )

            # Store the compressed weight differences
            update_id = (
                f"client_{self.node_id}_round_{round_context.round_number}_delta"
            )
            stored_id = self.storage_provider.put(update_id, compressed_delta_bytes)

            if stored_id:
                print(
                    f"FederatedClient {self.node_id}: Compressed ΔW stored at '{stored_id}'."
                )

                # Log transaction on blockchain
                tx_payload = {
                    "action": "client_delta_submission",
                    "client_id": self.node_id,
                    "round_number": round_context.round_number,
                    "storage_ref": stored_id,
                    "data_size": self.data_size,
                    "compression_ratio": compression_ratio,
                    "delta_hash": "mock_hash_of_delta",  # In real scenario, hash the delta
                }
                self.blockchain_connector.submit_transaction(tx_payload)
                print(
                    f"FederatedClient {self.node_id}: Transaction logged on blockchain for ΔW submission in round {round_context.round_number}."
                )
                return stored_id
            else:
                print(f"FederatedClient {self.node_id}: Failed to store compressed ΔW.")
                return None

        except Exception as e:
            print(f"FederatedClient {self.node_id}: Error sending ΔW update: {e}")
            return None

    def _compute_weight_difference(
        self, updated_weights: ModelWeights, initial_weights: ModelWeights
    ) -> ModelWeights:
        """
        Computes the difference between updated weights and initial weights (ΔW = W_updated - W_initial).
        This method should be implemented based on the specific ModelWeights format.
        For now, we assume ModelWeights is a structure that supports subtraction.
        """
        # This is a placeholder implementation
        # In a real implementation, this would depend on the ModelWeights format
        # For PyTorch tensors, this would be tensor subtraction
        # For numpy arrays, this would be array subtraction

        if hasattr(updated_weights, "__sub__"):
            # If ModelWeights supports direct subtraction
            return updated_weights - initial_weights
        elif isinstance(updated_weights, (list, tuple)):
            # If ModelWeights is a list/tuple of tensors/arrays
            return [u - i for u, i in zip(updated_weights, initial_weights)]
        elif isinstance(updated_weights, dict):
            # If ModelWeights is a dictionary of tensors/arrays
            return {
                key: updated_weights[key] - initial_weights[key]
                for key in updated_weights.keys()
            }
        else:
            # Fallback: return updated weights (equivalent to no compression)
            print(
                f"FederatedClient {self.node_id}: Warning - cannot compute ΔW for weights type {type(updated_weights)}, returning full weights."
            )
            return updated_weights

    def _copy_weights(self, weights: ModelWeights) -> ModelWeights:
        """
        Creates a deep copy of the model weights.
        """
        if hasattr(weights, "clone"):
            # PyTorch tensor
            return weights.clone()
        elif hasattr(weights, "copy"):
            # NumPy array
            return weights.copy()
        elif isinstance(weights, (list, tuple)):
            # List/tuple of tensors/arrays
            return [self._copy_weights(w) for w in weights]
        elif isinstance(weights, dict):
            # Dictionary of tensors/arrays
            return {key: self._copy_weights(value) for key, value in weights.items()}
        else:
            # Fallback: try to use copy module
            import copy

            return copy.deepcopy(weights)
