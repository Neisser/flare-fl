"""
Flare Builder Pattern Example - Clean and Modular Configuration

This example demonstrates how to use the Builder pattern to create
Orchestrators and Clients with a fluent, readable API.
"""

import numpy as np
import torch
import torch.nn as nn

# Flare imports using the new builder pattern
from flare import (
    ClientBuilder,
    FedAvg,
    InMemoryStorageProvider,
    MockChainConnector,
    NoCompression,
    OrchestratorBuilder,
    PowerSGDCompressor,
)

# Model imports (we'll move these to flare.models.examples later)
from flare.models.pytorch_adapter import PyTorchModelAdapter


class SimpleMLP(nn.Module):
    """Simple MLP for demonstration."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def generate_synthetic_data(num_samples=1000, input_size=784, num_classes=10):
    """Generate synthetic training data."""
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y


def create_client_datasets(num_clients=5, samples_per_client=200):
    """Create datasets for multiple clients."""
    datasets = []
    for i in range(num_clients):
        X, y = generate_synthetic_data(samples_per_client)
        datasets.append((X, y))
    return datasets


def demo_basic_orchestrator():
    """Demo: Basic Orchestrator with Builder pattern."""
    print("\n🔧 Demo: Basic Orchestrator with Builder")
    print("=" * 50)

    # Shared storage for all components
    storage = InMemoryStorageProvider()

    # Create orchestrator using builder pattern - much cleaner!
    orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=4))
        .with_storage_provider(storage)
        .with_blockchain(MockChainConnector())
        .with_aggregation_strategy(FedAvg())
        .with_rounds(num_rounds=2, clients_per_round=3)
        .with_client_training_params(epochs=2, learning_rate=0.01)
        .build()
    )

    print(f"✅ Created orchestrator: {type(orchestrator).__name__}")
    print(
        f"   - Model: {type(orchestrator.config.get('model_adapter').model).__name__}"
    )
    print(f"   - Compressor: {type(orchestrator.config.get('compressor')).__name__}")
    print(
        f"   - Aggregation: {type(orchestrator.config.get('aggregation_strategy')).__name__}"
    )

    return orchestrator, storage


def demo_mi_orchestrator():
    """Demo: MI Orchestrator for robust aggregation."""
    print("\n🛡️  Demo: MI Orchestrator with Malicious Detection")
    print("=" * 50)

    storage = InMemoryStorageProvider()

    # MI Orchestrator with automatic robust aggregation
    orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=4))
        .with_storage_provider(storage)
        .with_blockchain(MockChainConnector())
        .with_mi_settings(mi_threshold=0.1, min_clients=2, test_data_size=100)
        .with_rounds(num_rounds=2, clients_per_round=4)
        .build()
    )

    print(f"✅ Created MI orchestrator: {type(orchestrator).__name__}")
    print(f"   - MI threshold: {orchestrator.config.get('mi_threshold')}")
    print(f"   - Min clients: {orchestrator.config.get('min_clients')}")
    print(
        f"   - Aggregation: {type(orchestrator.config.get('aggregation_strategy')).__name__}"
    )

    return orchestrator, storage


def demo_vrf_orchestrator():
    """Demo: VRF Orchestrator with consensus validation."""
    print("\n🎲 Demo: VRF Orchestrator with Consensus")
    print("=" * 50)

    storage = InMemoryStorageProvider()

    # VRF Orchestrator with committee-based validation
    orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=4))
        .with_storage_provider(storage)
        .with_blockchain(MockChainConnector())
        .with_vrf_settings(
            committee_size=5, min_committee_threshold=0.6, vrf_seed="demo"
        )
        .with_rounds(num_rounds=2, clients_per_round=6)
        .build()
    )

    print(f"✅ Created VRF orchestrator: {type(orchestrator).__name__}")
    print(f"   - Committee size: {orchestrator.config.get('committee_size')}")
    print(f"   - Threshold: {orchestrator.config.get('min_committee_threshold')}")
    print(
        f"   - Consensus: {type(orchestrator.config.get('consensus_mechanism')).__name__}"
    )

    return orchestrator, storage


def demo_clients_creation(storage):
    """Demo: Creating multiple clients with Builder pattern."""
    print("\n👥 Demo: Creating Clients with Builder")
    print("=" * 50)

    client_datasets = create_client_datasets(num_clients=5, samples_per_client=200)
    clients = []

    for i, (X, y) in enumerate(client_datasets):
        # Each client can have different configurations
        compressor = PowerSGDCompressor(rank=4) if i % 2 == 0 else NoCompression()

        client = (
            ClientBuilder()
            .with_id(f"client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(compressor)
            .with_storage_provider(storage)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=2, learning_rate=0.01)
            .as_federated_client()  # Use FederatedClient with compression
            .build()
        )

        clients.append(client)
        print(f"✅ Created client {client.node_id}: {type(client).__name__}")
        print(f"   - Compressor: {type(client.config.get('compressor')).__name__}")
        print(f"   - Data size: {len(X)} samples")

    return clients


def demo_configuration_flexibility():
    """Demo: Builder flexibility and configuration options."""
    print("\n⚙️  Demo: Builder Flexibility")
    print("=" * 50)

    # Example 1: Minimal configuration (with defaults)
    simple_orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_storage_provider(InMemoryStorageProvider())
        .build()  # Uses defaults for everything else
    )

    print("✅ Simple orchestrator with defaults:")
    print(
        f"   - Compressor: {type(simple_orchestrator.config.get('compressor')).__name__}"
    )
    print(f"   - Rounds: {simple_orchestrator.config.get('num_rounds')}")

    # Example 2: Highly customized configuration
    custom_orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=8, power_iterations=2))
        .with_storage_provider(InMemoryStorageProvider())
        .with_blockchain(MockChainConnector())
        .with_aggregation_strategy(FedAvg())
        .with_rounds(num_rounds=5, clients_per_round=10)
        .with_client_training_params(epochs=3, learning_rate=0.001)
        .build()
    )

    print("\n✅ Custom orchestrator:")
    print(f"   - Compressor rank: {custom_orchestrator.config.get('compressor').rank}")
    print(f"   - Rounds: {custom_orchestrator.config.get('num_rounds')}")
    print(f"   - Learning rate: {custom_orchestrator.config.get('learning_rate')}")


def main():
    """Run all builder pattern demos."""
    print("🌟 Flare Builder Pattern Demo")
    print("=" * 50)
    print("This demo shows how the Builder pattern makes Flare")
    print("configuration much cleaner and more intuitive!")

    try:
        # Demo different orchestrator types
        basic_orch, storage1 = demo_basic_orchestrator()
        mi_orch, storage2 = demo_mi_orchestrator()
        vrf_orch, storage3 = demo_vrf_orchestrator()

        # Demo client creation
        clients = demo_clients_creation(storage1)
        print(f"\n📊 Created {len(clients)} clients successfully")

        # Demo configuration flexibility
        demo_configuration_flexibility()

        print("\n🎉 All Builder Pattern Demos Completed Successfully!")
        print("\n📝 Key Benefits:")
        print("   ✅ Fluent, readable configuration")
        print("   ✅ Automatic defaults for optional components")
        print("   ✅ Type-specific builders (basic/MI/VRF)")
        print("   ✅ Validation and error handling")
        print("   ✅ Easy extension for new phases")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
