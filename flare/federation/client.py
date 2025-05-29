from typing import Optional

from flare.blockchain import BlockchainConnector, TransactionPayload
from flare.compression import Compressor
from flare.core import FlareConfig, FlareNode, RoundContext
from flare.models import Metrics, ModelAdapter, ModelWeights, TrainData
from flare.models.adapters import EvalData
from flare.storage import StorageIdentifier, StorageProvider


class Client(FlareNode):
    """
    Represents a client device participating in federated learning.
    It trains a local model, compresses updates, and interacts with storage and blockchain.
    """

    def __init__(
        self,
        client_id: str,
        local_data: TrainData,  # Simplified for now, could be a DataLoader
        config: FlareConfig,
    ):
        super().__init__(client_id, config)
        self.local_data = local_data

        # Calculate data size properly for different data formats
        if isinstance(local_data, tuple) and len(local_data) == 2:
            # For (X, y) format, use the size of X
            X, y = local_data
            if hasattr(X, "shape"):
                self.data_size = X.shape[0]  # Number of samples
            else:
                self.data_size = len(X) if hasattr(X, "__len__") else 1
        else:
            # For other formats, try to get length
            self.data_size = len(local_data) if hasattr(local_data, "__len__") else 1

        # Initialize components from config
        self.model_adapter: ModelAdapter = config.get_required("model_adapter")
        self.compressor: Compressor = config.get_required("compressor")
        self.storage_provider: StorageProvider = config.get_required("storage_provider")
        self.blockchain_connector: BlockchainConnector = config.get_required(
            "blockchain_connector"
        )

        self.current_global_model_weights: Optional[ModelWeights] = None
        print(f"Client {self.node_id} initialized.")

    def start(self):
        """Starts the client's operation (e.g., listening for rounds)."""
        print(f"Client {self.node_id} started.")
        # In a real scenario, this might involve setting up a listener.

    def stop(self):
        """Stops the client and cleans up resources."""
        print(f"Client {self.node_id} stopped.")

    def receive_global_model(self, model_ref: StorageIdentifier) -> bool:
        """
        Receives the global model reference (e.g., a storage ID) and retrieves the model.
        """
        print(f"Client {self.node_id}: Receiving global model reference '{model_ref}'.")
        model_bytes = self.storage_provider.get(model_ref)
        if model_bytes is None:
            print(
                f"Client {self.node_id}: Failed to retrieve global model from '{model_ref}'."
            )
            return False

        # Decompress if necessary (though model serialization might not be compressed directly)
        # For now, assume model_bytes are directly deserializable by the adapter.
        # If model_bytes were compressed, we'd need to decompress them first.
        # For simplicity, we'll assume the model_adapter can handle the raw bytes.
        # Or, more correctly, the orchestrator stores *uncompressed* model bytes
        # and only *updates* are compressed. Let's assume updates are compressed.
        # So, for the global model, we deserialize directly.

        # Deserialize the full model (architecture + weights)
        try:
            self.model_adapter.deserialize_model(model_bytes)
            self.current_global_model_weights = self.model_adapter.get_weights()
            print(
                f"Client {self.node_id}: Global model received and loaded successfully."
            )
            return True
        except Exception as e:
            print(f"Client {self.node_id}: Error deserializing global model: {e}")
            return False

    def train_local(
        self, round_context: RoundContext, epochs: int, learning_rate: float
    ) -> Optional[ModelWeights]:
        """
        Performs local training on the client's data.
        Returns the updated local model weights.
        """
        if self.current_global_model_weights is None:
            print(f"Client {self.node_id}: Cannot train, no global model received yet.")
            return None

        print(
            f"Client {self.node_id}: Starting local training for round {round_context.round_number}..."
        )
        try:
            # Set the global model weights to start local training from
            self.model_adapter.set_weights(self.current_global_model_weights)

            # Perform local training
            train_history = self.model_adapter.train(
                data=self.local_data, epochs=epochs, learning_rate=learning_rate
            )
            print(
                f"Client {self.node_id}: Local training complete. History: {train_history}"
            )

            # Get the updated weights
            updated_weights = self.model_adapter.get_weights()
            return updated_weights
        except Exception as e:
            print(f"Client {self.node_id}: Error during local training: {e}")
            return None

    def send_update(
        self, round_context: RoundContext, updated_weights: ModelWeights
    ) -> Optional[StorageIdentifier]:
        """
        Compresses the updated weights and sends them to the storage provider.
        Also logs a transaction on the blockchain.
        """
        if updated_weights is None:
            print(f"Client {self.node_id}: No updated weights to send.")
            return None

        print(f"Client {self.node_id}: Compressing model update...")
        try:
            # Serialize weights to bytes first
            weights_bytes = self.model_adapter.serialize_weights()
            compressed_update_bytes = self.compressor.compress(weights_bytes)
            print(
                f"Client {self.node_id}: Update compressed from {len(weights_bytes)} to {len(compressed_update_bytes)} bytes."
            )

            # Store the compressed update
            update_id = (
                f"client_{self.node_id}_round_{round_context.round_number}_update"
            )
            stored_id = self.storage_provider.put(update_id, compressed_update_bytes)

            if stored_id:
                print(f"Client {self.node_id}: Update stored at '{stored_id}'.")
                # Log transaction on blockchain
                tx_payload: TransactionPayload = {
                    "action": "client_update_submission",
                    "client_id": self.node_id,
                    "round_number": round_context.round_number,
                    "storage_ref": stored_id,
                    "data_size": self.data_size,  # Important for weighted aggregation
                    "model_hash": "mock_hash_of_update",  # In a real scenario, hash the update
                }
                self.blockchain_connector.submit_transaction(tx_payload)
                print(
                    f"Client {self.node_id}: Transaction logged on blockchain for round {round_context.round_number}."
                )
                return stored_id
            else:
                print(f"Client {self.node_id}: Failed to store update.")
                return None
        except Exception as e:
            print(f"Client {self.node_id}: Error sending update: {e}")
            return None

    def evaluate_local_model(self, eval_data: EvalData) -> Metrics:
        """Evaluates the current local model (after training) on local data."""
        print(f"Client {self.node_id}: Evaluating local model...")
        metrics = self.model_adapter.evaluate(eval_data)
        print(f"Client {self.node_id}: Local evaluation metrics: {metrics}")
        return metrics
