from typing import Dict, List, Optional

from flare.models import ModelWeights

from .strategies import AggregationStrategy


class FedAvg(AggregationStrategy):
    """Federated Averaging strategy."""
    def __init__(self):
        print("FedAvg aggregation strategy initialized.")

    def aggregate(
        self,
        local_model_updates: List[ModelWeights],
        client_data_sizes: Optional[List[int]] = None,
        previous_global_weights: Optional[ModelWeights] = None
    ) -> ModelWeights:
        print(f"FedAvg: Aggregating {len(local_model_updates)} model updates.")

        if not local_model_updates:
            print("FedAvg: No updates to aggregate.")
            return previous_global_weights if previous_global_weights is not None else {}

        # For this mock, its assumed that weights are dictionaries of numbers.
        # A real implementation would need to handle NumPy arrays, PyTorch/TF tensors, etc.
        # and perform element-wise operations.
        num_clients = len(local_model_updates)
        aggregated_weights: Dict[str, float] = {}  # Assuming weights as dict[str, float] for the mock.

        # Determine aggregation weights based on client data sizes if provided.
        if client_data_sizes and len(client_data_sizes) == num_clients:
            total_data_size = sum(client_data_sizes)
            # Avoid division by zero if all sizes are 0
            if total_data_size == 0:
                aggregation_weights = [1.0 / num_clients] * num_clients
            else:
                aggregation_weights = [size / total_data_size for size in client_data_sizes]
            print(f"FedAvg: Using weighted averaging with weights: {aggregation_weights}")
        else:
            aggregation_weights = [1.0 / num_clients] * num_clients
            print("FedAvg: Using simple averaging (equal weights).")

        # Sum average weights for each key in the model updates.
        first_update_keys = None
        if isinstance(local_model_updates[0], dict):
            first_update_keys = local_model_updates[0].keys()

        if first_update_keys:  # If the weights are dictionaries.
            for key in first_update_keys:
                weighted_sum = 0.0
                for i, update in enumerate(local_model_updates):
                    if isinstance(update, dict) and key in update:
                        weighted_sum += update[key] * aggregation_weights[i]
                    else:
                        # Manage the case where an update does not have the expected key
                        # or is not a dictionary. it could raise an error or ignore it.
                        print(f"FedAvg: Warning - Key '{key}' not found or incompatible type in update {i}.")
                aggregated_weights[key] = weighted_sum
        else:
            # Here will be the logic for arrays/tensors.
            # For now, if they are not dictionaries, we simply return the first one (incorrect but it's a mock).
            print("FedAvg: Warning - Model updates are not dictionaries. Returning the first update as mock aggregation.")
            return local_model_updates[0]

        print(f"FedAvg: Aggregation complete. Result: {aggregated_weights}")
        return aggregated_weights
