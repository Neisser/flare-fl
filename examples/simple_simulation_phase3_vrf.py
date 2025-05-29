#!/usr/bin/env python3
"""
PHASE 3 Simulation: VRF Consensus with Committee-based Validation

This simulation demonstrates:
- VRF committee selection for model validation
- Committee-based consensus on aggregated models
- Decentralized decision making in federated learning
- Byzantine fault tolerance through voting

Features tested:
- Verifiable Random Function (VRF) for fair committee selection
- Committee member validation of aggregated models
- Consensus threshold-based approval/rejection
- Integration with existing FL pipeline (FASE 1 + 2 + 3)
"""

import numpy as np
import torch
import torch.nn as nn

from flare import (
    FederatedClient,
    FlareConfig,
    InMemoryStorageProvider,
    MIAggregationStrategy,
    MockChainConnector,
    PowerSGDCompressor,
    VRFConsensus,
)
from flare.models.pytorch_adapter import PyTorchModelAdapter


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for MNIST-like data."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MaliciousClient(FederatedClient):
    """Malicious client that may interfere with the consensus process."""

    def __init__(
        self,
        client_id: str,
        local_data,
        config: FlareConfig,
        malicious_type: str = "noise",
    ):
        super().__init__(client_id, local_data, config)
        self.malicious_type = malicious_type
        print(f"MaliciousClient {client_id} initialized with type: {malicious_type}")


def generate_synthetic_data(
    num_samples: int = 200, num_features: int = 784, num_classes: int = 10
):
    """Generate synthetic data for testing."""
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)

    # Convert to PyTorch tensors with correct data types
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).long()  # Convert to Long for classification

    return X_tensor, y_tensor


def create_vrf_fl_setup():
    """Create VRF-enabled federated learning setup."""
    print("🔧 Setting up VRF-enabled Federated Learning...")

    # Create global model
    global_model = SimpleMLP()
    global_adapter = PyTorchModelAdapter(global_model)

    # Initialize shared storage and blockchain
    storage = InMemoryStorageProvider()
    blockchain = MockChainConnector()

    # Create VRF consensus mechanism
    vrf_consensus = VRFConsensus(
        committee_size=3,  # Smaller committee for demo
        min_committee_threshold=0.6,  # 60% agreement needed
        vrf_seed="phase3_demo_seed",  # For reproducible results
    )

    # Setup configuration
    config = FlareConfig()
    config.set("model_adapter", global_adapter)
    config.set("storage_provider", storage)
    config.set("blockchain_connector", blockchain)
    config.set("compressor", PowerSGDCompressor(rank=8, power_iterations=1))

    # MI aggregation strategy (from Phase 2)
    aggregation_strategy = MIAggregationStrategy(
        mi_threshold=0.15, min_clients=2, test_data_size=50
    )
    config.set("aggregation_strategy", aggregation_strategy)

    return global_adapter, storage, blockchain, vrf_consensus, config


def create_diverse_clients(
    config: FlareConfig, num_honest: int = 5, num_malicious: int = 2
):
    """Create a mix of honest and malicious clients."""
    clients = []

    # Create honest clients
    for i in range(num_honest):
        client_id = f"honest_client_{i + 1}"
        local_data = generate_synthetic_data(num_samples=150)
        client = FederatedClient(client_id, local_data, config)
        clients.append(client)

    # Create malicious clients
    malicious_types = ["noise", "random"]
    for i in range(num_malicious):
        client_id = f"malicious_client_{i + 1}"
        local_data = generate_synthetic_data(num_samples=100)
        malicious_type = malicious_types[i % len(malicious_types)]
        client = MaliciousClient(client_id, local_data, config, malicious_type)
        clients.append(client)

    return clients


def test_vrf_committee_selection():
    """Test VRF committee selection mechanism."""
    print("\n🧪 Test 1: VRF Committee Selection")

    vrf = VRFConsensus(committee_size=3, vrf_seed="test_seed")
    available_nodes = [f"node_{i}" for i in range(7)]

    # Test deterministic selection
    committee1 = vrf.select_committee(available_nodes, round_number=1)
    committee2 = vrf.select_committee(available_nodes, round_number=1)
    committee3 = vrf.select_committee(available_nodes, round_number=2)

    print(f"Round 1 Committee (attempt 1): {committee1}")
    print(f"Round 1 Committee (attempt 2): {committee2}")
    print(f"Round 2 Committee: {committee3}")

    # Verify deterministic selection
    assert committee1 == committee2, "VRF should be deterministic for same round"
    assert len(committee1) == 3, "Committee size should be 3"
    assert len(set(committee1)) == 3, "Committee members should be unique"

    print("✅ VRF committee selection works correctly")
    return True


def test_consensus_voting():
    """Test committee voting and consensus mechanism."""
    print("\n🧪 Test 2: Committee Voting and Consensus")

    vrf = VRFConsensus(committee_size=3, min_committee_threshold=0.6)
    committee = ["validator_1", "validator_2", "validator_3"]
    vrf.current_committee = committee

    # Test proposal creation
    proposal_data = {
        "model_hash": "test_hash_123",
        "round_number": 1,
        "validation_type": "test",
    }

    proposal_id = vrf.propose_decision(proposal_data, "test_proposer")
    print(f"Created proposal: {proposal_id}")

    # Test voting scenarios
    # Scenario 1: Majority approval (2/3 = 66.7% > 60%)
    vrf.vote(proposal_id, "validator_1", True, {"score": 0.9})
    vrf.vote(proposal_id, "validator_2", True, {"score": 0.8})
    vrf.vote(proposal_id, "validator_3", False, {"score": 0.5})

    result = vrf.get_consensus_result(proposal_id)
    print(f"Consensus result: {result['result'] if result else 'pending'}")

    assert result is not None, "Consensus result should not be None"
    assert result["result"] == "approved", "Should be approved with 2/3 votes"

    print("✅ Committee voting and consensus works correctly")
    return True


def test_full_vrf_fl_round():
    """Test complete VRF-enabled federated learning round."""
    print("\n🧪 Test 3: Complete VRF-FL Round")

    # Setup
    global_adapter, storage, blockchain, vrf_consensus, config = create_vrf_fl_setup()
    clients = create_diverse_clients(config, num_honest=5, num_malicious=2)

    # Store initial global model
    initial_weights = global_adapter.get_weights()
    global_model_id = "global_model_v1"
    # Serialize the model before storing
    serialized_model = global_adapter.serialize_model()
    storage.put(global_model_id, serialized_model)

    # Create test dataset for MI aggregation
    test_data = generate_synthetic_data(num_samples=100)

    print(f"Created {len(clients)} clients (5 honest + 2 malicious)")

    # Simulate VRF-enabled federated round
    round_number = 1
    client_ids = [client.node_id for client in clients]

    # Phase 1: VRF Committee Selection
    print("\n--- Phase 1: VRF Committee Selection ---")
    committee = vrf_consensus.select_committee(
        available_nodes=client_ids, round_number=round_number
    )
    print(f"Selected committee: {committee}")

    # Phase 1.5: Distribute global model to all clients
    print("\n--- Phase 1.5: Distribute Global Model ---")
    for client in clients:
        success = client.receive_global_model(global_model_id)
        if not success:
            print(f"Failed to distribute global model to {client.node_id}")
            return False

    # Phase 2: Local Training (honest and malicious)
    print("\n--- Phase 2: Local Training ---")
    local_updates = {}
    data_sizes = {}

    for client in clients:
        print(f"Training client: {client.node_id}")

        # Create round context
        from flare.core import RoundContext

        round_context = RoundContext(
            round_number=round_number,
            global_model_version=global_model_id,
            metadata={"committee": committee},
        )

        # Local training
        try:
            delta_weights = client.train_local(
                round_context=round_context, epochs=2, learning_rate=0.01
            )

            if delta_weights is not None:
                local_updates[client.node_id] = delta_weights

                # Get data size
                if hasattr(client.local_data, "__len__"):
                    data_sizes[client.node_id] = len(client.local_data)
                else:
                    data_sizes[client.node_id] = client.local_data[0].shape[0]
            else:
                print(f"Client {client.node_id} returned None for delta_weights")

        except Exception as e:
            print(f"Error training {client.node_id}: {e}")
            continue

    print(f"Collected updates from {len(local_updates)} clients")

    if len(local_updates) == 0:
        print("❌ No valid updates collected, cannot proceed with aggregation")
        return False

    # Phase 3: MI-based Aggregation (from Phase 2)
    print("\n--- Phase 3: MI-based Aggregation ---")
    mi_strategy = config.get("aggregation_strategy")

    try:
        # Convert dictionaries to lists for MI aggregation
        update_list = list(local_updates.values())
        size_list = list(data_sizes.values())

        aggregated_weights = mi_strategy.aggregate(
            local_model_updates=update_list,
            client_data_sizes=size_list,
            previous_global_weights=initial_weights,
            test_data=test_data,
        )
        print("✅ MI aggregation completed successfully")

    except Exception as e:
        print(f"❌ MI aggregation failed: {e}")
        return False

    # Phase 4: VRF Committee Validation
    print("\n--- Phase 4: VRF Committee Validation ---")

    # Create validation proposal
    import hashlib
    import pickle

    model_hash = hashlib.sha256(pickle.dumps(aggregated_weights)).hexdigest()[:16]

    proposal_data = {
        "round_number": round_number,
        "model_hash": model_hash,
        "validation_type": "aggregated_model",
        "aggregation_method": "MI_filtered",
    }

    proposal_id = vrf_consensus.propose_decision(proposal_data, "orchestrator")
    print(f"Created validation proposal: {proposal_id}")

    # Simulate committee validation
    validation_results = {}
    for member_id in committee:
        # Simulate validation score (normally done by committee members)
        import random

        base_score = 0.75
        variation = (random.random() - 0.5) * 0.3  # ±15%
        validation_score = max(0.0, min(1.0, base_score + variation))

        # Vote based on score
        vote = validation_score > 0.7

        vote_accepted = vrf_consensus.vote(
            proposal_id=proposal_id,
            voter_id=member_id,
            vote=vote,
            vote_data={"validation_score": validation_score},
        )

        validation_results[member_id] = {
            "score": validation_score,
            "vote": vote,
            "accepted": vote_accepted,
        }

        print(f"{member_id}: score={validation_score:.3f}, vote={vote}")

    # Get consensus result
    consensus_result = vrf_consensus.get_consensus_result(proposal_id)
    if consensus_result:
        print(f"\nConsensus result: {consensus_result['result']}")
    else:
        print("\nConsensus result: pending")
        return False

    # Phase 5: Update Global Model (if approved)
    if consensus_result["result"] == "approved":
        print("\n--- Phase 5: Global Model Update ---")
        # Store new global model
        new_global_id = f"global_model_v{round_number + 1}"
        # Serialize the updated model before storing
        global_adapter.set_weights(aggregated_weights)
        serialized_updated_model = global_adapter.serialize_model()
        storage.put(new_global_id, serialized_updated_model)

        # Log to blockchain (simplified - just print for demo)
        blockchain_entry = {
            "round": round_number,
            "proposal_id": proposal_id,
            "committee": committee,
            "consensus": consensus_result["result"],
            "model_id": new_global_id,
            "model_hash": model_hash,
        }
        print(f"Blockchain entry: {blockchain_entry}")

        print(f"✅ Global model updated and stored as {new_global_id}")
    else:
        print("❌ Global model update rejected by committee")

    return consensus_result["result"] == "approved"


def run_multi_round_vrf_simulation():
    """Run multi-round VRF simulation."""
    print("\n🎯 Multi-Round VRF Simulation")

    # Setup
    global_adapter, storage, blockchain, vrf_consensus, config = create_vrf_fl_setup()
    clients = create_diverse_clients(config, num_honest=6, num_malicious=1)
    test_data = generate_synthetic_data(num_samples=100)

    # Store initial model
    current_weights = global_adapter.get_weights()
    # Serialize the model before storing
    serialized_model = global_adapter.serialize_model()
    storage.put("global_model_v0", serialized_model)

    results = []

    for round_num in range(1, 4):  # 3 rounds
        print(f"\n{'=' * 50}")
        print(f"🔄 VRF Round {round_num}")
        print(f"{'=' * 50}")

        # Committee selection
        client_ids = [client.node_id for client in clients]
        committee = vrf_consensus.select_committee(client_ids, round_num)

        # Create round context
        from flare.core import RoundContext

        round_context = RoundContext(
            round_number=round_num,
            global_model_version=f"global_model_v{round_num - 1}",
            metadata={"committee": committee, "vrf_seed": vrf_consensus.vrf_seed},
        )

        # Collect local updates
        local_updates = {}
        data_sizes = {}

        # Distribute current global model to all clients
        current_model_id = f"global_model_v{round_num - 1}"
        for client in clients:
            client.receive_global_model(current_model_id)

        for client in clients:
            try:
                delta_weights = client.train_local(
                    round_context, epochs=1, learning_rate=0.01
                )

                if delta_weights is not None:
                    local_updates[client.node_id] = delta_weights

                    if hasattr(client.local_data, "__len__"):
                        data_sizes[client.node_id] = len(client.local_data)
                    else:
                        data_sizes[client.node_id] = client.local_data[0].shape[0]

            except Exception as e:
                print(f"Client {client.node_id} failed: {e}")

        # Skip round if no valid updates
        if len(local_updates) == 0:
            print(f"Round {round_num}: No valid updates, skipping...")
            continue

        # Aggregate with MI filtering
        mi_strategy = config.get("aggregation_strategy")

        # Convert dictionaries to lists for MI aggregation
        update_list = list(local_updates.values())
        size_list = list(data_sizes.values())

        aggregated_weights = mi_strategy.aggregate(
            local_model_updates=update_list,
            client_data_sizes=size_list,
            previous_global_weights=current_weights,
            test_data=test_data,
        )

        # Committee validation
        import hashlib
        import pickle

        model_hash = hashlib.sha256(pickle.dumps(aggregated_weights)).hexdigest()[:16]

        proposal_id = vrf_consensus.propose_decision(
            {
                "round_number": round_num,
                "model_hash": model_hash,
                "validation_type": "aggregated_model",
            },
            "orchestrator",
        )

        # Committee voting
        for member_id in committee:
            import random

            score = 0.8 + (random.random() - 0.5) * 0.2  # 0.7-0.9 range
            vote = score > 0.75
            vrf_consensus.vote(proposal_id, member_id, vote, {"score": score})

        # Get result
        consensus_result = vrf_consensus.get_consensus_result(proposal_id)
        approved = consensus_result and consensus_result.get("result") == "approved"

        if approved:
            current_weights = aggregated_weights
            # Serialize the updated model before storing
            global_adapter.set_weights(current_weights)
            serialized_model = global_adapter.serialize_model()
            storage.put(f"global_model_v{round_num}", serialized_model)

        round_result = {
            "round": round_num,
            "committee": committee,
            "approved": approved,
            "consensus": consensus_result,
        }
        results.append(round_result)

        print(f"Round {round_num}: {'✅ APPROVED' if approved else '❌ REJECTED'}")

    # Final statistics
    print("\n📊 VRF Simulation Results")
    print(f"{'=' * 50}")
    total_rounds = len(results)
    approved_rounds = sum(1 for r in results if r["approved"])

    print(f"Total rounds: {total_rounds}")
    print(f"Approved rounds: {approved_rounds}")
    if total_rounds > 0:
        print(f"Approval rate: {approved_rounds / total_rounds:.1%}")
    else:
        print("Approval rate: N/A (no completed rounds)")

    # VRF consensus stats
    vrf_stats = vrf_consensus.get_consensus_stats()
    print("\nVRF Consensus Statistics:")
    print(f"Total proposals: {vrf_stats['total_proposals']}")
    print(f"Approved proposals: {vrf_stats['approved']}")
    if "committee_size" in vrf_stats:
        print(f"Committee size: {vrf_stats['committee_size']}")
    else:
        print(f"Committee size: {vrf_consensus.committee_size}")

    return results


def main():
    """Run all Phase 3 tests and simulations."""
    print("🚀 FLARE PHASE 3: VRF Consensus Implementation")
    print("=" * 60)

    try:
        # Test 1: VRF Committee Selection
        test1_passed = test_vrf_committee_selection()

        # Test 2: Consensus Voting
        test2_passed = test_consensus_voting()

        # Test 3: Full VRF-FL Round
        test3_passed = test_full_vrf_fl_round()

        # Multi-round simulation
        simulation_results = run_multi_round_vrf_simulation()

        # Final summary
        print("\n🎉 PHASE 3 COMPLETION SUMMARY")
        print(f"{'=' * 60}")
        print(f"✅ VRF Committee Selection: {'PASS' if test1_passed else 'FAIL'}")
        print(f"✅ Consensus Voting: {'PASS' if test2_passed else 'FAIL'}")
        print(f"✅ Full VRF-FL Round: {'PASS' if test3_passed else 'FAIL'}")
        print(f"✅ Multi-round Simulation: {len(simulation_results)} rounds completed")

        all_tests_passed = test1_passed and test2_passed and test3_passed

        if all_tests_passed:
            print("\n🎊 ALL PHASE 3 TESTS PASSED!")
            print("✅ VRF consensus successfully integrated with FL pipeline")
            print("✅ Committee-based validation working correctly")
            print("✅ Decentralized decision making functional")
            return True
        else:
            print("\n❌ Some Phase 3 tests failed")
            return False

    except Exception as e:
        print(f"\n❌ Phase 3 simulation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
