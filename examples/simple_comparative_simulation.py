"""
Simple Comparative Simulation - Builder Pattern Showcase

This simulation demonstrates the Builder pattern flexibility by running
multiple federated learning scenarios with different configurations.
"""

import time

import numpy as np
import torch
import torch.nn as nn

# Flare imports using Builder pattern
from flare import (
    ClientBuilder,
    FedAvg,
    InMemoryStorageProvider,
    MockChainConnector,
    NoCompression,
    OrchestratorBuilder,
    PowerSGDCompressor,
)

# Model imports
from flare.models.pytorch_adapter import PyTorchModelAdapter


class SimpleMLP(nn.Module):
    """Simple MLP for demonstration."""

    def __init__(self, input_size=784, hidden_size=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def generate_simple_data(num_samples=200, input_size=784, num_classes=10):
    """Generate simple synthetic data as PyTorch tensors."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def create_client_datasets(num_clients=4, samples_per_client=150):
    """Create simple datasets for clients."""
    datasets = []
    for i in range(num_clients):
        X, y = generate_simple_data(samples_per_client)
        datasets.append((X, y))
    return datasets


def run_basic_scenario():
    """Run basic FL scenario with Builder pattern."""
    print("\n🔧 BASIC FL Scenario")
    print("-" * 40)

    # Shared storage
    storage = InMemoryStorageProvider()

    # Create evaluation data
    eval_X, eval_y = generate_simple_data(100)
    eval_data = (eval_X, eval_y)

    # Create orchestrator with Builder - clean and simple!
    orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(NoCompression())
        .with_storage_provider(storage)
        .with_blockchain(MockChainConnector())
        .with_aggregation_strategy(FedAvg())
        .with_rounds(num_rounds=2, clients_per_round=3)
        .with_client_training_params(epochs=1, learning_rate=0.01)
        .with_eval_data(eval_data)
        .build()
    )

    # Create clients with Builder
    client_datasets = create_client_datasets(num_clients=4, samples_per_client=100)
    clients = []

    for i, (X, y) in enumerate(client_datasets):
        client = (
            ClientBuilder()
            .with_id(f"basic_client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(NoCompression())
            .with_storage_provider(storage)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=1, learning_rate=0.01)
            .as_federated_client()
            .build()
        )
        clients.append(client)

    print(f"✅ Created orchestrator: {type(orchestrator).__name__}")
    print(f"✅ Created {len(clients)} clients")

    # Run simulation
    start_time = time.time()

    # Register clients and run
    orchestrator.register_clients(clients)

    # Get initial performance
    initial_metrics = orchestrator.evaluate_global_model(eval_data)
    print(f"📊 Initial accuracy: {initial_metrics.get('accuracy', 0.0):.4f}")

    # Run FL rounds
    orchestrator.start()

    # Get final performance
    final_metrics = orchestrator.evaluate_global_model(eval_data)
    duration = time.time() - start_time

    print(f"📊 Final accuracy: {final_metrics.get('accuracy', 0.0):.4f}")
    print(f"⏱️  Duration: {duration:.2f}s")

    return {
        "name": "Basic FL",
        "initial_accuracy": initial_metrics.get("accuracy", 0.0),
        "final_accuracy": final_metrics.get("accuracy", 0.0),
        "duration": duration,
        "orchestrator_type": type(orchestrator).__name__,
        "compressor_type": type(orchestrator.config.get("compressor")).__name__,
    }


def run_compressed_scenario():
    """Run compressed FL scenario with PowerSGD."""
    print("\n🗜️  COMPRESSED FL Scenario")
    print("-" * 40)

    # Shared storage
    storage = InMemoryStorageProvider()

    # Create evaluation data
    eval_X, eval_y = generate_simple_data(100)
    eval_data = (eval_X, eval_y)

    # Create orchestrator with compression - Builder makes this easy!
    orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=4, power_iterations=1))
        .with_storage_provider(storage)
        .with_blockchain(MockChainConnector())
        .with_aggregation_strategy(FedAvg())
        .with_rounds(num_rounds=2, clients_per_round=3)
        .with_client_training_params(epochs=1, learning_rate=0.01)
        .with_eval_data(eval_data)
        .build()
    )

    # Create clients with different compression strategies
    client_datasets = create_client_datasets(num_clients=4, samples_per_client=100)
    clients = []

    compression_options = [
        PowerSGDCompressor(rank=4),
        PowerSGDCompressor(rank=8),
        NoCompression(),
        PowerSGDCompressor(rank=2),
    ]

    for i, (X, y) in enumerate(client_datasets):
        compressor = compression_options[i % len(compression_options)]

        client = (
            ClientBuilder()
            .with_id(f"compressed_client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(compressor)
            .with_storage_provider(storage)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=1, learning_rate=0.01)
            .as_federated_client()
            .build()
        )
        clients.append(client)
        print(f"   Client {i + 1}: {type(compressor).__name__}")

    print(f"✅ Created orchestrator: {type(orchestrator).__name__}")
    print(f"✅ Created {len(clients)} clients with mixed compression")

    # Run simulation
    start_time = time.time()

    # Register clients and run
    orchestrator.register_clients(clients)

    # Get initial performance
    initial_metrics = orchestrator.evaluate_global_model(eval_data)
    print(f"📊 Initial accuracy: {initial_metrics.get('accuracy', 0.0):.4f}")

    # Run FL rounds
    orchestrator.start()

    # Get final performance
    final_metrics = orchestrator.evaluate_global_model(eval_data)
    duration = time.time() - start_time

    print(f"📊 Final accuracy: {final_metrics.get('accuracy', 0.0):.4f}")
    print(f"⏱️  Duration: {duration:.2f}s")

    return {
        "name": "Compressed FL",
        "initial_accuracy": initial_metrics.get("accuracy", 0.0),
        "final_accuracy": final_metrics.get("accuracy", 0.0),
        "duration": duration,
        "orchestrator_type": type(orchestrator).__name__,
        "compressor_type": type(orchestrator.config.get("compressor")).__name__,
    }


def run_mi_scenario():
    """Run MI-based robust FL scenario."""
    print("\n🛡️  ROBUST FL (MI) Scenario")
    print("-" * 40)

    # Shared storage
    storage = InMemoryStorageProvider()

    # Create evaluation data
    eval_X, eval_y = generate_simple_data(100)
    eval_data = (eval_X, eval_y)

    # Create MI orchestrator - Builder automatically detects MI type!
    orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=6))
        .with_storage_provider(storage)
        .with_blockchain(MockChainConnector())
        .with_mi_settings(mi_threshold=0.1, min_clients=2, test_data_size=50)
        .with_rounds(num_rounds=2, clients_per_round=3)
        .with_client_training_params(epochs=1, learning_rate=0.01)
        .with_eval_data(eval_data)
        .build()
    )

    # Create honest clients
    client_datasets = create_client_datasets(num_clients=4, samples_per_client=100)
    clients = []

    for i, (X, y) in enumerate(client_datasets):
        client = (
            ClientBuilder()
            .with_id(f"mi_client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(PowerSGDCompressor(rank=6))
            .with_storage_provider(storage)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=1, learning_rate=0.01)
            .as_federated_client()
            .build()
        )
        clients.append(client)

    print(f"✅ Created orchestrator: {type(orchestrator).__name__}")
    print(f"✅ Created {len(clients)} robust clients")

    # Run simulation
    start_time = time.time()

    # Register clients and run
    orchestrator.register_clients(clients)

    # Get initial performance
    initial_metrics = orchestrator.evaluate_global_model(eval_data)
    print(f"📊 Initial accuracy: {initial_metrics.get('accuracy', 0.0):.4f}")

    # Run FL rounds
    orchestrator.start()

    # Get final performance
    final_metrics = orchestrator.evaluate_global_model(eval_data)
    duration = time.time() - start_time

    print(f"📊 Final accuracy: {final_metrics.get('accuracy', 0.0):.4f}")
    print(f"⏱️  Duration: {duration:.2f}s")

    return {
        "name": "Robust FL (MI)",
        "initial_accuracy": initial_metrics.get("accuracy", 0.0),
        "final_accuracy": final_metrics.get("accuracy", 0.0),
        "duration": duration,
        "orchestrator_type": type(orchestrator).__name__,
        "compressor_type": type(orchestrator.config.get("compressor")).__name__,
    }


def print_comparison_results(results):
    """Print comparison table of all scenarios."""
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE RESULTS - BUILDER PATTERN SHOWCASE")
    print("=" * 80)

    print("\n📈 Performance Summary:")
    print("-" * 80)
    print(
        f"{'Scenario':<20} {'Type':<15} {'Compressor':<15} {'Initial':<10} {'Final':<10} {'Improve':<10} {'Time':<8}"
    )
    print("-" * 80)

    for result in results:
        improvement = result["final_accuracy"] - result["initial_accuracy"]
        print(
            f"{result['name']:<20} "
            f"{result['orchestrator_type']:<15} "
            f"{result['compressor_type']:<15} "
            f"{result['initial_accuracy']:<10.4f} "
            f"{result['final_accuracy']:<10.4f} "
            f"{improvement:<10.4f} "
            f"{result['duration']:<8.2f}s"
        )

    print("-" * 80)

    # Analysis
    best_accuracy = max(results, key=lambda x: x["final_accuracy"])
    best_improvement = max(
        results, key=lambda x: x["final_accuracy"] - x["initial_accuracy"]
    )
    fastest = min(results, key=lambda x: x["duration"])

    print(f"\n🔍 Analysis:")
    print(
        f"🏆 Best Final Accuracy: {best_accuracy['name']} ({best_accuracy['final_accuracy']:.4f})"
    )
    print(
        f"📈 Best Improvement: {best_improvement['name']} (+{best_improvement['final_accuracy'] - best_improvement['initial_accuracy']:.4f})"
    )
    print(f"⚡ Fastest: {fastest['name']} ({fastest['duration']:.2f}s)")


def main():
    """Run the comparative simulation showcasing Builder pattern."""
    print("🌟 FLARE BUILDER PATTERN COMPARATIVE SIMULATION")
    print("=" * 60)
    print("Demonstrating Builder flexibility with different FL configurations:")
    print("• Basic FL (minimal config)")
    print("• Compressed FL (PowerSGD with mixed client compression)")
    print("• Robust FL (MI-based aggregation)")
    print("=" * 60)

    try:
        # Run all scenarios
        results = []

        # Each scenario demonstrates different Builder usage
        results.append(run_basic_scenario())
        results.append(run_compressed_scenario())
        results.append(run_mi_scenario())

        # Print comparative results
        print_comparison_results(results)

        print("\n🎉 Simulation Completed Successfully!")
        print("\n💡 Builder Pattern Benefits Demonstrated:")
        print("   ✅ Fluent, readable configuration syntax")
        print("   ✅ Automatic orchestrator type selection (MI)")
        print("   ✅ Easy component switching (compression, aggregation)")
        print("   ✅ Consistent interface across all scenarios")
        print("   ✅ Reduced boilerplate code")
        print("   ✅ Clear separation of configuration vs execution")

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
