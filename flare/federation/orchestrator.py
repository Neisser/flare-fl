# flare/federation/orchestrator.py
import random
from typing import Any, Dict, List, Optional

from flare.blockchain import BlockchainConnector, ConsensusMechanism, TransactionPayload
from flare.compression import Compressor
from flare.core import FlareConfig, FlareNode, RoundContext
from flare.federation.client import Client  # Import the Client class
from flare.federation.strategies import AggregationStrategy
from flare.models import EvalData, Metrics, ModelAdapter, ModelWeights
from flare.storage import StorageIdentifier, StorageProvider


class Orchestrator(FlareNode):
    """
    Coordinates the federated learning process, including client selection,
    model distribution, update collection, aggregation, and blockchain interaction.
    """
    def __init__(self, config: FlareConfig):
        super().__init__("orchestrator_node", config)

        # Initialize components from config
        self.model_adapter: ModelAdapter = config.get_required('model_adapter')
        self.compressor: Compressor = config.get_required('compressor')
        self.storage_provider: StorageProvider = config.get_required('storage_provider')
        self.blockchain_connector: BlockchainConnector = config.get_required('blockchain_connector')
        self.consensus_mechanism: ConsensusMechanism = config.get_required('consensus_mechanism')
        self.aggregation_strategy: AggregationStrategy = config.get_required('aggregation_strategy')

        self.registered_clients: List[Client] = []
        self.current_round_number: int = 0
        self.global_model_weights: ModelWeights = self.model_adapter.get_weights()  # Initial global model
        self.num_rounds: int = config.get('num_rounds', 1)
        self.clients_per_round: int = config.get('clients_per_round', 1)

        print("Orchestrator initialized with initial global model.")
        # Initial blockchain log for the genesis model
        self._log_global_model_on_blockchain(self.global_model_weights, 0)

    def start(self):
        """Starts the orchestrator's main loop."""
        print("Orchestrator started. Beginning federated training rounds.")
        for _ in range(self.num_rounds):
            self.execute_round()
        print("Federated training complete.")

    def stop(self):
        """Stops the orchestrator and cleans up resources."""
        print("Orchestrator stopped.")

    def register_client(self, client: Client):
        """Registers a client with the orchestrator."""
        if client not in self.registered_clients:
            self.registered_clients.append(client)
            print(f"Orchestrator: Client {client.node_id} registered.")
        else:
            print(f"Orchestrator: Client {client.node_id} already registered.")

    def register_clients(self, clients: List[Client]):
        """Registers multiple clients."""
        for client in clients:
            self.register_client(client)

    def select_clients(self) -> List[Client]:
        """Selects a subset of clients for the current round."""
        if not self.registered_clients:
            print("Orchestrator: No clients registered for selection.")
            return []

        num_to_select = min(self.clients_per_round, len(self.registered_clients))
        selected_clients = random.sample(self.registered_clients, num_to_select)
        print(f"Orchestrator: Selected {len(selected_clients)} clients for round {self.current_round_number + 1}.")
        return selected_clients

    def _distribute_global_model(self, round_context: RoundContext, selected_clients: List[Client]) -> Optional[StorageIdentifier]:
        """
        Serializes and stores the global model, then sends its reference to clients.
        """
        print(f"Orchestrator: Distributing global model for round {round_context.round_number}...")
        # Serialize the full model (architecture + weights)
        model_bytes = self.model_adapter.serialize_model()
        model_ref_id = f"global_model_round_{round_context.round_number}"
        stored_id = self.storage_provider.put(model_ref_id, model_bytes)

        if stored_id:
            print(f"Orchestrator: Global model stored at '{stored_id}'. Notifying clients.")
            # Notify clients to fetch the model
            for client in selected_clients:
                client.receive_global_model(stored_id)
            return stored_id
        else:
            print("Orchestrator: Failed to store global model for distribution.")
            return None

    def _collect_updates(self, round_context: RoundContext, selected_clients: List[Client]) -> List[Dict[str, Any]]:
        """
        Collects model updates from selected clients.
        In a real system, this would involve waiting for client submissions.
        For simulation, we'll directly call client methods.
        """
        print(f"Orchestrator: Collecting updates from {len(selected_clients)} clients...")
        collected_updates: List[Dict[str, Any]] = []
        for client in selected_clients:
            print(f"Orchestrator: Requesting update from client {client.node_id}...")
            # Client trains locally
            updated_weights = client.train_local(
                round_context=round_context,
                epochs=self.config.get('client_epochs', 1),
                learning_rate=self.config.get('client_learning_rate', 0.01)
            )
            # Client sends update (which also stores and logs it)
            storage_ref = client.send_update(round_context, updated_weights)

            if storage_ref:
                # Retrieve compressed update from storage
                compressed_bytes = self.storage_provider.get(storage_ref)
                if compressed_bytes:
                    # Decompress and deserialize weights
                    decompressed_bytes = self.compressor.decompress(compressed_bytes)
                    client_weights = self.model_adapter.deserialize_weights(decompressed_bytes)

                    # Validate contribution using consensus mechanism
                    if self.consensus_mechanism.validate_contribution(client_weights, client_id=client.node_id):
                        collected_updates.append({
                            "client_id": client.node_id,
                            "weights": client_weights,
                            "data_size": client.data_size  # Pass data size for weighted aggregation
                        })
                        print(f"Orchestrator: Collected valid update from client {client.node_id}.")
                    else:
                        print(f"Orchestrator: Client {client.node_id} update failed consensus validation. Skipping.")
                else:
                    print(f"Orchestrator: Failed to retrieve compressed update for client {client.node_id}.")
            else:
                print(f"Orchestrator: Client {client.node_id} failed to send update.")
        return collected_updates

    def _aggregate_updates(self, collected_updates: List[Dict[str, Any]]) -> Optional[ModelWeights]:
        """Aggregates the collected model updates."""
        if not collected_updates:
            print("Orchestrator: No updates to aggregate.")
            return None

        updates_only = [u['weights'] for u in collected_updates]
        data_sizes = [u['data_size'] for u in collected_updates]

        print(f"Orchestrator: Aggregating {len(updates_only)} updates using {type(self.aggregation_strategy).__name__}.")
        new_global_weights = self.aggregation_strategy.aggregate(
            local_model_updates=updates_only,
            client_data_sizes=data_sizes,
            previous_global_weights=self.global_model_weights
        )
        return new_global_weights

    def _log_global_model_on_blockchain(self, model_weights: ModelWeights, round_number: int):
        """Logs the new global model's hash/reference on the blockchain."""
        # In a real scenario, you'd calculate a cryptographic hash of the model weights
        # or the model's storage reference (e.g., IPFS CID).
        # For this mock, we'll use a simple string representation.
        model_hash = f"hash_of_global_model_round_{round_number}"  # Placeholder
        # If using IPFS, this would be the CID of the model stored by the orchestrator.

        tx_payload: TransactionPayload = {
            "action": "new_global_model_registered",
            "round_number": round_number,
            "model_hash": model_hash,
            # "storage_ref": "..." if the model itself is stored on IPFS by orchestrator
        }
        self.blockchain_connector.submit_transaction(tx_payload)
        print(f"Orchestrator: Global model for round {round_number} logged on blockchain with hash: {model_hash}.")

    def evaluate_global_model(self, eval_data: EvalData) -> Metrics:
        """Evaluates the current global model."""
        print("Orchestrator: Evaluating current global model...")
        self.model_adapter.set_weights(self.global_model_weights)  # Ensure adapter has current global weights
        metrics = self.model_adapter.evaluate(eval_data)
        print(f"Orchestrator: Global model evaluation metrics: {metrics}")
        return metrics

    def execute_round(self):
        """Executes a single round of federated learning."""
        self.current_round_number += 1
        print(f"\n--- Orchestrator: Starting Round {self.current_round_number} ---")
        round_context = RoundContext(self.current_round_number, self.global_model_weights)

        # 1. Client Selection
        selected_clients = self.select_clients()
        if not selected_clients:
            print("Orchestrator: No clients selected. Ending round.")
            return

        # 2. Global Model Distribution
        global_model_storage_ref = self._distribute_global_model(round_context, selected_clients)
        if not global_model_storage_ref:
            print("Orchestrator: Failed to distribute global model. Ending round.")
            return

        # 3. Collect Updates
        collected_updates = self._collect_updates(round_context, selected_clients)
        if not collected_updates:
            print("Orchestrator: No updates collected. Ending round.")
            return

        # 4. Aggregate Updates
        new_global_weights = self._aggregate_updates(collected_updates)
        if new_global_weights is None:
            print("Orchestrator: Aggregation failed. Ending round.")
            return

        # 5. Consensus and Update Global Model
        # The consensus mechanism here is about agreeing on the *result* of aggregation.
        # For MockPoAConsensus, it just accepts the aggregated result.
        agreed_global_weights = self.consensus_mechanism.reach_agreement(
            proposals=[new_global_weights],  # Pass the aggregated weights as the proposal
            round_context=round_context
        )

        if agreed_global_weights is not None:
            self.global_model_weights = agreed_global_weights
            self.model_adapter.set_weights(self.global_model_weights)  # Update the adapter's model
            print(f"Orchestrator: Global model updated for round {self.current_round_number}.")
            self._log_global_model_on_blockchain(self.global_model_weights, self.current_round_number)
        else:
            print(f"Orchestrator: Consensus failed for round {self.current_round_number}. Global model not updated.")

        print(f"--- Orchestrator: Round {self.current_round_number} Finished ---")

    def get_global_model(self) -> ModelAdapter:
        """Returns the current global model adapter instance."""
        self.model_adapter.set_weights(self.global_model_weights)  # Ensure it's up-to-date
        return self.model_adapter
