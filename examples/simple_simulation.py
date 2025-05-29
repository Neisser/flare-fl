import random
from typing import List

from flare.blockchain import MockChainConnector, MockPoAConsensus
from flare.compression import ZlibCompressor

# Import Flare components
from flare.core import FlareConfig
from flare.federation import Client, FedAvg, Orchestrator
from flare.models import EvalData, MockModelAdapter, TrainData
from flare.storage import InMemoryStorageProvider


# --- Utility for Mock Data (Simplified) ---
def load_mock_data_partitions(num_clients: int, data_per_client: int = 100) -> List[TrainData]:
    """
    Generates mock data for simulation.
    Each 'data' item is just a list of random numbers to simulate input features.
    """
    print(f"Generating mock data for {num_clients} clients, {data_per_client} items each.")
    partitions = []
    for i in range(num_clients):
        # Simulate some data that might change slightly per client
        # In a real scenario, this would load actual dataset splits
        mock_client_data = [random.random() for _ in range(data_per_client)]
        partitions.append(mock_client_data)
    return partitions


def create_mock_eval_data(size: int = 500) -> EvalData:
    """Creates mock evaluation data."""
    return [random.random() for _ in range(size)]


# --- Main Simulation Function ---
def run_simulation():
    print("--- Starting Flare Simulation ---")

    # 1. Configure Flare components
    config = FlareConfig()

    # Model: Using MockModelAdapter for now
    initial_mock_model = "MyInitialCNN"  # Just a string name for the mock model
    config.set('model_adapter', MockModelAdapter(initial_mock_model))

    # Compression: Choose a compressor
    config.set('compressor', ZlibCompressor(level=1))  # Using Zlib for demonstration
    # config.set('compressor', NoCompression()) # Or no compression

    # Storage: In-memory for simulation
    config.set('storage_provider', InMemoryStorageProvider())

    # Blockchain: Mock for simulation
    mock_blockchain_connector = MockChainConnector()
    config.set('blockchain_connector', mock_blockchain_connector)
    config.set('consensus_mechanism', MockPoAConsensus(config))

    # Aggregation Strategy
    config.set('aggregation_strategy', FedAvg())

    # Simulation parameters
    config.set('num_rounds', 3)
    config.set('total_clients', 10)
    config.set('clients_per_round', 3)  # Number of clients selected per round
    config.set('client_epochs', 2)
    config.set('client_learning_rate', 0.001)

    print("\n--- Configuration Summary ---")
    print(config)
    print("-----------------------------\n")

    # 2. Prepare Data and Initialize Clients
    total_clients = config.get_required('total_clients')
    mock_data_partitions = load_mock_data_partitions(total_clients)
    clients: List[Client] = []
    for i in range(total_clients):
        # Each client gets its own config copy to ensure isolated component instances if needed
        # For simplicity, we pass the same shared component instances from the main config
        client = Client(
            client_id=f"client_{i+1}",
            local_data=mock_data_partitions[i],
            config=config  # Pass the main config, components are shared by reference
        )
        clients.append(client)
        client.start()  # Start each client (e.g., to listen for orchestrator)

    # 3. Initialize Orchestrator
    orchestrator = Orchestrator(config)
    orchestrator.register_clients(clients)
    orchestrator.start()  # Start the orchestrator's training loop

    # 4. (Optional) Evaluate initial global model
    mock_eval_data = create_mock_eval_data()
    print("\n--- Initial Global Model Evaluation ---")
    orchestrator.evaluate_global_model(mock_eval_data)

    # The orchestrator.start() method now handles the rounds.
    # We can add a final evaluation after the loop.
    print("\n--- Final Global Model Evaluation ---")
    orchestrator.evaluate_global_model(mock_eval_data)

    # Stop all clients
    for client in clients:
        client.stop()

    orchestrator.stop()

    print("\n--- Blockchain Transaction Log (Mock) ---")
    for tx in mock_blockchain_connector.get_chain_log():
        print(f"Tx Hash: {tx['tx_hash']}, Block: {tx['block_number']}, Payload: {tx['payload']}")

    print("\n--- Mock Blockchain State ---")
    print(mock_blockchain_connector.get_current_state())

    print("\n--- Flare Simulation Finished ---")


if __name__ == "__main__":
    run_simulation()
