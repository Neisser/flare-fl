"""
Flare Comparative Simulation - Builder Pattern Showcase

This simulation demonstrates the Builder pattern flexibility by running
multiple federated learning scenarios with different configurations:
- Basic vs MI vs VRF Orchestrators
- Different compression strategies
- Various client configurations
- Performance comparison across scenarios
"""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Flare imports using Builder pattern
from flare import (
    ClientBuilder,
    FedAvg,
    GzipCompressor,
    InMemoryStorageProvider,
    MockChainConnector,
    NoCompression,
    OrchestratorBuilder,
    PowerSGDCompressor,
)

# Model imports
from flare.models.pytorch_adapter import PyTorchModelAdapter


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST-like classification."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def generate_realistic_data(
    num_samples=1000, input_size=784, num_classes=10, noise_level=0.1
):
    """Generate more realistic synthetic data with some structure."""
    # Create structured data (not completely random)
    X = np.random.randn(num_samples, input_size).astype(np.float32)

    # Add some structure to make classification meaningful
    class_centers = np.random.randn(num_classes, input_size) * 2
    y = np.random.randint(0, num_classes, num_samples)

    for i in range(num_samples):
        class_idx = y[i]
        X[i] = X[i] * noise_level + class_centers[class_idx]

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).long()  # CrossEntropyLoss expects long type

    return X_tensor, y_tensor


def create_heterogeneous_client_datasets(num_clients=8, base_samples=300):
    """Create heterogeneous datasets simulating real federated scenarios."""
    print(f"📊 Creating {num_clients} heterogeneous client datasets...")

    datasets = []
    data_sizes = []

    for i in range(num_clients):
        # Vary dataset sizes to simulate realistic federation
        size_multiplier = np.random.uniform(0.5, 2.0)  # 50% to 200% of base size
        client_samples = int(base_samples * size_multiplier)

        # Add data heterogeneity (some clients have more of certain classes)
        if i < 2:  # First 2 clients: more class 0-2
            noise_level = 0.05
            class_bias = [0, 1, 2]
        elif i < 4:  # Next 2 clients: more class 3-5
            noise_level = 0.1
            class_bias = [3, 4, 5]
        elif i < 6:  # Next 2 clients: more class 6-8
            noise_level = 0.15
            class_bias = [6, 7, 8]
        else:  # Last 2 clients: balanced but noisy
            noise_level = 0.2
            class_bias = list(range(10))

        X, y = generate_realistic_data(client_samples, noise_level=noise_level)

        # Apply class bias
        biased_indices = np.isin(y, class_bias)
        X = X[biased_indices]
        y = y[biased_indices]

        datasets.append((X, y))
        data_sizes.append(len(X))
        print(
            f"   Client {i + 1}: {len(X)} samples, classes {np.unique(y)}, noise={noise_level}"
        )

    return datasets, data_sizes


def create_evaluation_data(num_samples=1000):
    """Create balanced evaluation dataset."""
    X, y = generate_realistic_data(num_samples, noise_level=0.05)
    return X, y


class SimulationScenario:
    """Represents a complete simulation scenario with specific configuration."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.orchestrator = None
        self.clients = []
        self.results = {}
        self.start_time = None
        self.end_time = None

    def add_result(self, key: str, value: Any):
        """Add a result metric."""
        self.results[key] = value

    def get_duration(self) -> float:
        """Get simulation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def create_basic_scenario(
    storage_provider, client_datasets, eval_data
) -> SimulationScenario:
    """Create a basic FL scenario with minimal compression."""
    print("\n🔧 Setting up BASIC scenario...")

    scenario = SimulationScenario(
        "Basic FL",
        "Standard federated learning with basic aggregation and no compression",
    )

    # Basic orchestrator - minimal configuration
    scenario.orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(NoCompression())
        .with_storage_provider(storage_provider)
        .with_blockchain(MockChainConnector())
        .with_aggregation_strategy(FedAvg())
        .with_rounds(num_rounds=3, clients_per_round=6)
        .with_client_training_params(epochs=2, learning_rate=0.01)
        .with_eval_data(eval_data)
        .build()
    )

    # Create clients with basic configuration
    for i, (X, y) in enumerate(client_datasets):
        client = (
            ClientBuilder()
            .with_id(f"basic_client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(NoCompression())
            .with_storage_provider(storage_provider)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=2, learning_rate=0.01)
            .as_federated_client()
            .build()
        )
        scenario.clients.append(client)

    print(f"   ✅ Created {len(scenario.clients)} basic clients")
    return scenario


def create_compressed_scenario(
    storage_provider, client_datasets, eval_data
) -> SimulationScenario:
    """Create FL scenario with PowerSGD compression."""
    print("\n🗜️  Setting up COMPRESSED scenario...")

    scenario = SimulationScenario(
        "Compressed FL",
        "Federated learning with PowerSGD compression for efficient communication",
    )

    # Orchestrator with compression
    scenario.orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=8, power_iterations=2))
        .with_storage_provider(storage_provider)
        .with_blockchain(MockChainConnector())
        .with_aggregation_strategy(FedAvg())
        .with_rounds(num_rounds=3, clients_per_round=6)
        .with_client_training_params(epochs=2, learning_rate=0.01)
        .with_eval_data(eval_data)
        .build()
    )

    # Create clients with different compression strategies
    compression_strategies = [
        PowerSGDCompressor(rank=4),
        PowerSGDCompressor(rank=8),
        GzipCompressor(),
        NoCompression(),  # Some clients without compression
    ]

    for i, (X, y) in enumerate(client_datasets):
        # Cycle through compression strategies
        compressor = compression_strategies[i % len(compression_strategies)]

        client = (
            ClientBuilder()
            .with_id(f"compressed_client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(compressor)
            .with_storage_provider(storage_provider)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=2, learning_rate=0.01)
            .as_federated_client()
            .build()
        )
        scenario.clients.append(client)

    print(f"   ✅ Created {len(scenario.clients)} clients with mixed compression")
    return scenario


def create_robust_scenario(
    storage_provider, client_datasets, eval_data
) -> SimulationScenario:
    """Create FL scenario with MI-based robust aggregation."""
    print("\n🛡️  Setting up ROBUST (MI) scenario...")

    scenario = SimulationScenario(
        "Robust FL (MI)",
        "Federated learning with MI-based malicious detection and PowerSGD compression",
    )

    # MI Orchestrator with robust aggregation
    scenario.orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=6, power_iterations=1))
        .with_storage_provider(storage_provider)
        .with_blockchain(MockChainConnector())
        .with_mi_settings(mi_threshold=0.15, min_clients=4, test_data_size=200)
        .with_rounds(num_rounds=3, clients_per_round=6)
        .with_client_training_params(epochs=2, learning_rate=0.01)
        .with_eval_data(eval_data)
        .build()
    )

    # Create honest clients with good compression
    for i, (X, y) in enumerate(client_datasets):
        client = (
            ClientBuilder()
            .with_id(f"robust_client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(PowerSGDCompressor(rank=6))
            .with_storage_provider(storage_provider)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=2, learning_rate=0.01)
            .as_federated_client()
            .build()
        )
        scenario.clients.append(client)

    print(f"   ✅ Created {len(scenario.clients)} robust clients with MI detection")
    return scenario


def create_consensus_scenario(
    storage_provider, client_datasets, eval_data
) -> SimulationScenario:
    """Create FL scenario with VRF consensus validation."""
    print("\n🎲 Setting up CONSENSUS (VRF) scenario...")

    scenario = SimulationScenario(
        "Consensus FL (VRF)",
        "Federated learning with VRF committee consensus and full compression pipeline",
    )

    # VRF Orchestrator with consensus validation
    scenario.orchestrator = (
        OrchestratorBuilder()
        .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
        .with_compressor(PowerSGDCompressor(rank=8, power_iterations=2))
        .with_storage_provider(storage_provider)
        .with_blockchain(MockChainConnector())
        .with_aggregation_strategy(FedAvg())  # Will be wrapped by VRF logic
        .with_vrf_settings(
            committee_size=5,
            min_committee_threshold=0.6,
            vrf_seed="comparative_simulation",
        )
        .with_rounds(num_rounds=3, clients_per_round=8)  # Use all clients
        .with_client_training_params(epochs=2, learning_rate=0.01)
        .with_eval_data(eval_data)
        .build()
    )

    # Create high-quality clients for consensus scenario
    for i, (X, y) in enumerate(client_datasets):
        client = (
            ClientBuilder()
            .with_id(f"consensus_client_{i + 1}")
            .with_local_data((X, y))
            .with_model_adapter(PyTorchModelAdapter(SimpleMLP()))
            .with_compressor(PowerSGDCompressor(rank=8))
            .with_storage_provider(storage_provider)
            .with_blockchain_connector(MockChainConnector())
            .with_training_params(epochs=2, learning_rate=0.01)
            .as_federated_client()
            .build()
        )
        scenario.clients.append(client)

    print(
        f"   ✅ Created {len(scenario.clients)} consensus clients with VRF validation"
    )
    return scenario


def run_scenario(scenario: SimulationScenario) -> None:
    """Run a complete simulation scenario."""
    print(f"\n🚀 Running scenario: {scenario.name}")
    print(f"   📝 {scenario.description}")

    scenario.start_time = time.time()

    try:
        # Register all clients
        scenario.orchestrator.register_clients(scenario.clients)

        # Get initial model performance
        initial_metrics = scenario.orchestrator.evaluate_global_model(
            scenario.orchestrator.config.get("eval_data")
        )
        scenario.add_result("initial_accuracy", initial_metrics.get("accuracy", 0.0))

        # Run federated training
        print(
            f"   🔄 Starting {scenario.orchestrator.config.get('num_rounds')} FL rounds..."
        )
        scenario.orchestrator.start()

        # Get final model performance
        final_metrics = scenario.orchestrator.evaluate_global_model(
            scenario.orchestrator.config.get("eval_data")
        )
        scenario.add_result("final_accuracy", final_metrics.get("accuracy", 0.0))
        scenario.add_result(
            "accuracy_improvement",
            final_metrics.get("accuracy", 0.0) - initial_metrics.get("accuracy", 0.0),
        )

        # Calculate additional metrics
        scenario.add_result("num_clients", len(scenario.clients))
        scenario.add_result(
            "num_rounds", scenario.orchestrator.config.get("num_rounds")
        )
        scenario.add_result(
            "compressor_type",
            type(scenario.orchestrator.config.get("compressor")).__name__,
        )
        scenario.add_result("orchestrator_type", type(scenario.orchestrator).__name__)

        scenario.end_time = time.time()
        scenario.add_result("duration_seconds", scenario.get_duration())

        print(f"   ✅ Scenario completed successfully!")
        print(f"      - Duration: {scenario.get_duration():.2f}s")
        print(f"      - Initial accuracy: {scenario.results['initial_accuracy']:.4f}")
        print(f"      - Final accuracy: {scenario.results['final_accuracy']:.4f}")
        print(f"      - Improvement: {scenario.results['accuracy_improvement']:.4f}")

    except Exception as e:
        scenario.end_time = time.time()
        scenario.add_result("error", str(e))
        print(f"   ❌ Scenario failed: {e}")


def print_comparative_results(scenarios: List[SimulationScenario]) -> None:
    """Print a detailed comparison of all scenarios."""
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE SIMULATION RESULTS")
    print("=" * 80)

    # Results table
    print("\n📈 Performance Summary:")
    print("-" * 80)
    print(
        f"{'Scenario':<20} {'Type':<15} {'Compressor':<15} {'Initial':<10} {'Final':<10} {'Improve':<10} {'Time':<8}"
    )
    print("-" * 80)

    for scenario in scenarios:
        if "error" not in scenario.results:
            print(
                f"{scenario.name:<20} "
                f"{scenario.results.get('orchestrator_type', 'N/A'):<15} "
                f"{scenario.results.get('compressor_type', 'N/A'):<15} "
                f"{scenario.results.get('initial_accuracy', 0):<10.4f} "
                f"{scenario.results.get('final_accuracy', 0):<10.4f} "
                f"{scenario.results.get('accuracy_improvement', 0):<10.4f} "
                f"{scenario.results.get('duration_seconds', 0):<8.2f}s"
            )
        else:
            print(f"{scenario.name:<20} ERROR: {scenario.results['error']}")

    print("-" * 80)

    # Detailed analysis
    print("\n🔍 Detailed Analysis:")

    successful_scenarios = [s for s in scenarios if "error" not in s.results]

    if successful_scenarios:
        # Best performing scenario
        best_accuracy = max(
            successful_scenarios, key=lambda s: s.results.get("final_accuracy", 0)
        )
        best_improvement = max(
            successful_scenarios, key=lambda s: s.results.get("accuracy_improvement", 0)
        )
        fastest = min(
            successful_scenarios,
            key=lambda s: s.results.get("duration_seconds", float("inf")),
        )

        print(
            f"🏆 Best Final Accuracy: {best_accuracy.name} ({best_accuracy.results['final_accuracy']:.4f})"
        )
        print(
            f"📈 Best Improvement: {best_improvement.name} (+{best_improvement.results['accuracy_improvement']:.4f})"
        )
        print(
            f"⚡ Fastest Execution: {fastest.name} ({fastest.results['duration_seconds']:.2f}s)"
        )

        # Compression analysis
        print(f"\n💾 Compression Analysis:")
        for scenario in successful_scenarios:
            compressor = scenario.results.get("compressor_type", "Unknown")
            accuracy = scenario.results.get("final_accuracy", 0)
            print(f"   {compressor}: {accuracy:.4f} accuracy")

        # Orchestrator type analysis
        print(f"\n🎯 Orchestrator Type Analysis:")
        for scenario in successful_scenarios:
            orch_type = scenario.results.get("orchestrator_type", "Unknown")
            improvement = scenario.results.get("accuracy_improvement", 0)
            print(f"   {orch_type}: +{improvement:.4f} improvement")


def main():
    """Run the comprehensive comparative simulation."""
    print("🌟 FLARE COMPARATIVE SIMULATION")
    print("=" * 60)
    print("Testing Builder Pattern flexibility with multiple FL scenarios:")
    print("• Basic FL (minimal config)")
    print("• Compressed FL (PowerSGD + mixed compression)")
    print("• Robust FL (MI-based malicious detection)")
    print("• Consensus FL (VRF committee validation)")
    print("=" * 60)

    try:
        # Setup common components
        print("\n🔧 Setting up simulation environment...")
        storage_provider = InMemoryStorageProvider()
        client_datasets, data_sizes = create_heterogeneous_client_datasets(
            num_clients=8, base_samples=250
        )
        eval_data = create_evaluation_data(num_samples=500)

        print(f"✅ Environment ready:")
        print(f"   - {len(client_datasets)} client datasets")
        print(
            f"   - Total training samples: {sum(len(data[0]) for data in client_datasets)}"
        )
        print(f"   - Evaluation samples: {len(eval_data[0])}")

        # Create all scenarios using Builder pattern
        scenarios = [
            create_basic_scenario(storage_provider, client_datasets, eval_data),
            create_compressed_scenario(storage_provider, client_datasets, eval_data),
            create_robust_scenario(storage_provider, client_datasets, eval_data),
            create_consensus_scenario(storage_provider, client_datasets, eval_data),
        ]

        print(f"\n📋 Created {len(scenarios)} scenarios for comparison")

        # Run all scenarios
        for scenario in scenarios:
            run_scenario(scenario)

        # Print comparative results
        print_comparative_results(scenarios)

        print("\n🎉 Comparative Simulation Completed!")
        print("\n💡 Key Insights from Builder Pattern:")
        print("   ✅ Easy configuration switching between FL strategies")
        print("   ✅ Automatic component selection (MI/VRF orchestrators)")
        print("   ✅ Flexible client configurations with different compression")
        print("   ✅ Clean, readable scenario setup")
        print("   ✅ Consistent interface across all FL types")

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
