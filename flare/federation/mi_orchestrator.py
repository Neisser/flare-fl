import pickle
from typing import Any, Dict, List, Optional, Tuple

from flare.core import FlareConfig, RoundContext

from .mi_aggregation_strategy import MIAggregationStrategy
from .orchestrator import Orchestrator


class MIOrchestrator(Orchestrator):
    """
    Enhanced Orchestrator with Mutual Information-based aggregation.

    This orchestrator uses MIAggregationStrategy to detect and filter out
    potentially malicious clients based on mutual information analysis
    of their model outputs.
    """

    def __init__(self, config: FlareConfig, test_dataset: Optional[Tuple] = None):
        """
        Initialize MI-enhanced orchestrator.

        Args:
            config: Flare configuration
            test_dataset: Global test dataset for MI computation (X, y)
        """
        super().__init__(config)

        # Store test dataset for MI computation
        self.test_dataset = test_dataset

        # Use MI aggregation strategy by default
        if (
            not hasattr(config, "_config")
            or "aggregation_strategy" not in config._config
        ):
            self.aggregation_strategy = MIAggregationStrategy(
                mi_threshold=0.1, min_clients=2, test_data_size=100
            )
            config.set("aggregation_strategy", self.aggregation_strategy)

        # Track malicious detection history
        self.malicious_detection_history: Dict[int, List[str]] = {}

        print("MIOrchestrator initialized with MI-based malicious detection")

    def execute_round(self, round_num: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Execute a federated learning round with MI-based filtering.

        Args:
            round_num: Round number
            **kwargs: Additional parameters

        Returns:
            Round execution results including malicious detection info
        """
        print(f"\n=== MIOrchestrator: Executing Round {round_num} ===")

        # Create round context
        round_context = RoundContext(
            round_number=round_num,
            global_model_version=f"v{round_num}",
            metadata={"mi_filtering": True},
        )

        # Select clients for this round
        selected_clients = self.select_clients()
        if not selected_clients:
            print("MIOrchestrator: No clients available for training")
            return {"error": "No clients available"}

        print(
            f"MIOrchestrator: Selected {len(selected_clients)} clients for round {round_num}"
        )

        # Step 1: Distribute current global model
        print("1. Distributing global model to clients...")
        global_model_ref = self._distribute_global_model()
        if not global_model_ref:
            print("MIOrchestrator: Failed to distribute global model")
            return {"error": "Failed to distribute global model"}

        # Step 2: Clients receive model and train locally
        print("2. Clients training locally...")
        client_updates = []
        client_data_sizes = []
        failed_clients = []

        for client in selected_clients:
            try:
                # Client receives global model
                success = client.receive_global_model(global_model_ref)
                if not success:
                    print(
                        f"MIOrchestrator: Client {client.node_id} failed to receive global model"
                    )
                    failed_clients.append(client.node_id)
                    continue

                # Client trains locally
                delta_weights = client.train_local(
                    round_context,
                    epochs=kwargs.get("epochs", 3),
                    learning_rate=kwargs.get("learning_rate", 0.01),
                )

                if delta_weights is None:
                    print(f"MIOrchestrator: Client {client.node_id} failed to train")
                    failed_clients.append(client.node_id)
                    continue

                # Store update reference
                update_ref = client.send_update(round_context, delta_weights)
                if update_ref:
                    client_updates.append(update_ref)
                    # Get data size for weighted aggregation
                    data_size = self._get_client_data_size(client)
                    client_data_sizes.append(data_size)
                else:
                    print(
                        f"MIOrchestrator: Client {client.node_id} failed to send update"
                    )
                    failed_clients.append(client.node_id)

            except Exception as e:
                print(f"MIOrchestrator: Error with client {client.node_id}: {e}")
                failed_clients.append(client.node_id)

        if not client_updates:
            print("MIOrchestrator: No valid client updates received")
            return {"error": "No valid client updates"}

        print(
            f"MIOrchestrator: Collected {len(client_updates)} updates, {len(failed_clients)} clients failed"
        )

        # Step 3: Retrieve and decompress client updates
        print("3. Retrieving and processing client updates...")
        decompressed_updates = []
        valid_data_sizes = []

        for i, update_ref in enumerate(client_updates):
            try:
                # Retrieve compressed update from storage
                compressed_update = self.storage_provider.get(update_ref)
                if compressed_update is None:
                    print(f"MIOrchestrator: Failed to retrieve update {i}")
                    continue

                # Decompress the update (we need the first client's compressor)
                # In practice, the orchestrator might have its own decompressor
                client = selected_clients[i]  # Assuming same order
                decompressed_data = client.compressor.decompress(compressed_update)

                # Deserialize the weights
                delta_weights = pickle.loads(decompressed_data)
                decompressed_updates.append(delta_weights)
                valid_data_sizes.append(client_data_sizes[i])

            except Exception as e:
                print(f"MIOrchestrator: Error processing update {i}: {e}")
                continue

        if not decompressed_updates:
            print("MIOrchestrator: No valid decompressed updates")
            return {"error": "No valid decompressed updates"}

        # Step 4: Apply MI-based aggregation with malicious detection
        print("4. Aggregating updates with MI-based filtering...")

        # Get current global weights for aggregation
        current_global_weights = self.model_adapter.get_weights()

        # Prepare aggregation parameters
        aggregation_kwargs = {}
        if self.test_dataset is not None:
            aggregation_kwargs["test_data"] = self.test_dataset

        # Use MI aggregation strategy
        aggregated_weights = self.aggregation_strategy.aggregate(
            local_model_updates=decompressed_updates,
            client_data_sizes=valid_data_sizes,
            previous_global_weights=current_global_weights,
            **aggregation_kwargs,
        )

        # Step 5: Update global model
        print("5. Updating global model...")
        try:
            self.model_adapter.set_weights(aggregated_weights)

            # Store updated global model
            updated_model_bytes = self.model_adapter.serialize_model()
            new_global_ref = self.storage_provider.put(
                f"global_model_round_{round_num + 1}", updated_model_bytes
            )

            # Log round completion to blockchain
            round_result = {
                "round_number": round_num,
                "participating_clients": len(decompressed_updates),
                "failed_clients": len(failed_clients),
                "global_model_ref": new_global_ref,
                "mi_filtering_enabled": True,
            }

            tx_receipt = self.blockchain_connector.submit_transaction(round_result)

            print(f"MIOrchestrator: Round {round_num} completed successfully")
            print(f"MIOrchestrator: {len(decompressed_updates)} clients participated")
            print(
                f"MIOrchestrator: Global model updated and stored at {new_global_ref}"
            )

            return {
                "success": True,
                "round_number": round_num,
                "participating_clients": len(decompressed_updates),
                "failed_clients": failed_clients,
                "global_model_ref": new_global_ref,
                "transaction_receipt": tx_receipt,
                "mi_filtering_applied": True,
            }

        except Exception as e:
            print(f"MIOrchestrator: Error updating global model: {e}")
            return {"error": f"Failed to update global model: {e}"}

    def _distribute_global_model(self) -> Optional[str]:
        """
        Serialize and store current global model for distribution.

        Returns:
            Storage reference for the global model
        """
        try:
            # Serialize current global model
            global_model_bytes = self.model_adapter.serialize_model()

            # Store in shared storage
            storage_ref = self.storage_provider.put(
                "global_model_current", global_model_bytes
            )

            return storage_ref

        except Exception as e:
            print(f"MIOrchestrator: Error distributing global model: {e}")
            return None

    def _get_client_data_size(self, client) -> int:
        """
        Get the size of a client's local dataset.

        Args:
            client: The client instance

        Returns:
            Number of training samples
        """
        try:
            if hasattr(client, "local_data") and client.local_data is not None:
                X, y = client.local_data
                if hasattr(X, "shape"):
                    return X.shape[0]
                elif hasattr(X, "__len__"):
                    return len(X)
            return 100  # Default size if unknown
        except Exception:
            return 100  # Default size on error

    def get_malicious_detection_report(self) -> Dict[str, Any]:
        """
        Get a report of malicious detection activity across rounds.

        Returns:
            Report containing malicious detection statistics
        """
        return {
            "detection_history": self.malicious_detection_history,
            "total_rounds": len(self.malicious_detection_history),
            "mi_threshold": getattr(self.aggregation_strategy, "mi_threshold", "N/A"),
            "strategy_type": type(self.aggregation_strategy).__name__,
        }
