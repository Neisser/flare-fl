from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from flare.models import ModelWeights

from .strategies import AggregationStrategy


class MIAggregationStrategy(AggregationStrategy):
    """
    Mutual Information-based Aggregation Strategy for robust federated learning.

    This strategy computes mutual information between model outputs to detect
    and filter out potentially malicious or anomalous contributions before aggregation.

    Algorithm:
    1. Collect model outputs from all clients on a test dataset
    2. Compute pairwise mutual information between outputs
    3. Identify outliers using MI threshold
    4. Aggregate only trusted (non-outlier) model updates
    """

    def __init__(
        self, mi_threshold: float = 0.1, min_clients: int = 2, test_data_size: int = 100
    ):
        """
        Initialize MI-based aggregation strategy.

        Args:
            mi_threshold: Minimum MI threshold for considering models similar
            min_clients: Minimum number of clients needed for aggregation
            test_data_size: Size of test dataset for MI computation
        """
        self.mi_threshold = mi_threshold
        self.min_clients = min_clients
        self.test_data_size = test_data_size

        # Store test data for MI computation (will be set during aggregation)
        self.test_features: Optional[np.ndarray] = None

        print(
            f"MIAggregationStrategy initialized with MI threshold={mi_threshold}, "
            f"min_clients={min_clients}"
        )

    def aggregate(
        self,
        local_model_updates: List[ModelWeights],
        client_data_sizes: Optional[List[int]] = None,
        previous_global_weights: Optional[ModelWeights] = None,
        **kwargs,
    ) -> ModelWeights:
        """
        Aggregate model updates using Mutual Information filtering.

        Args:
            local_model_updates: List of model weight updates from clients
            client_data_sizes: List of data sizes for weighted aggregation
            previous_global_weights: Previous global model weights
            **kwargs: Additional parameters (may include test_data for MI computation)

        Returns:
            Aggregated model weights from trusted clients only
        """
        print(
            f"MIAggregationStrategy: Aggregating {len(local_model_updates)} model updates"
        )

        if len(local_model_updates) < self.min_clients:
            print(
                f"MIAggregationStrategy: Not enough clients ({len(local_model_updates)} < {self.min_clients})"
            )
            return (
                previous_global_weights
                if previous_global_weights is not None
                else local_model_updates[0]
            )

        # Extract test data from kwargs if available
        test_data = kwargs.get("test_data", None)

        if test_data is not None:
            # Perform MI-based filtering
            trusted_indices = self._filter_malicious_updates(
                local_model_updates, test_data, previous_global_weights
            )

            if len(trusted_indices) < self.min_clients:
                print(
                    f"MIAggregationStrategy: Warning - Only {len(trusted_indices)} trusted clients found, "
                    f"using all clients for aggregation"
                )
                trusted_indices = list(range(len(local_model_updates)))
        else:
            print("MIAggregationStrategy: No test data provided, skipping MI filtering")
            trusted_indices = list(range(len(local_model_updates)))

        # Select trusted updates and corresponding data sizes
        trusted_updates = [local_model_updates[i] for i in trusted_indices]
        trusted_data_sizes = (
            [client_data_sizes[i] for i in trusted_indices]
            if client_data_sizes
            else None
        )

        print(
            f"MIAggregationStrategy: Aggregating {len(trusted_updates)} trusted updates "
            f"(filtered out {len(local_model_updates) - len(trusted_updates)} potentially malicious updates)"
        )

        # Perform weighted aggregation of trusted updates
        return self._weighted_average(
            trusted_updates, trusted_data_sizes, previous_global_weights
        )

    def _filter_malicious_updates(
        self,
        model_updates: List[ModelWeights],
        test_data: Tuple[np.ndarray, np.ndarray],
        global_weights: Optional[ModelWeights],
    ) -> List[int]:
        """
        Filter out malicious model updates using Mutual Information analysis.

        Args:
            model_updates: List of model weight updates
            test_data: Test dataset (X, y) for MI computation
            global_weights: Current global model weights for baseline

        Returns:
            List of indices of trusted model updates
        """
        print(
            "MIAggregationStrategy: Computing mutual information for malicious detection..."
        )

        X_test, y_test = test_data

        # Limit test data size for efficiency
        if len(X_test) > self.test_data_size:
            indices = np.random.choice(len(X_test), self.test_data_size, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]

        # Compute model outputs for MI calculation
        # For simplicity in simulation, we'll use a hash-based approach
        # In production, this would involve actual model inference
        model_signatures = []

        for i, weights in enumerate(model_updates):
            # Create a signature from model weights for MI computation
            signature = self._compute_model_signature(weights, X_test)
            model_signatures.append(signature)

        # Compute pairwise MI between model signatures
        mi_matrix = self._compute_pairwise_mi(model_signatures)

        # Identify trusted models based on MI similarity
        trusted_indices = self._identify_trusted_models(mi_matrix)

        return trusted_indices

    def _compute_model_signature(
        self, weights: ModelWeights, test_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute a signature for a model based on its weights.

        This is a simplified approach for simulation. In production, this would
        involve running the actual model on test data to get predictions.

        Args:
            weights: Model weights
            test_features: Test input features

        Returns:
            Model signature as a numpy array
        """
        # Simplified signature: combine weight statistics with pseudo-predictions
        signature_components = []

        # Extract statistics from weights
        if isinstance(weights, dict):
            for name, weight_tensor in weights.items():
                if hasattr(weight_tensor, "detach"):
                    # PyTorch tensor
                    weight_array = weight_tensor.detach().cpu().numpy()
                else:
                    weight_array = np.array(weight_tensor)

                # Compute statistics: mean, std, min, max
                stats = [
                    np.mean(weight_array),
                    np.std(weight_array),
                    np.min(weight_array),
                    np.max(weight_array),
                ]
                signature_components.extend(stats)

        # Add pseudo-predictions based on simple linear combination
        # This simulates what the model's output distribution might look like
        if len(signature_components) > 0 and len(test_features) > 0:
            # Create a simple linear predictor using weight statistics
            weight_summary = np.array(
                signature_components[: min(10, len(signature_components))]
            )

            # Pseudo-predictions as linear combination of test features and weight summary
            if len(weight_summary) > 0 and test_features.shape[1] > 0:
                # Take first few features and combine with weight statistics
                n_features = min(test_features.shape[1], len(weight_summary))
                pseudo_preds = np.dot(
                    test_features[:, :n_features], weight_summary[:n_features]
                )

                # Normalize and add to signature
                pseudo_preds = (pseudo_preds - np.mean(pseudo_preds)) / (
                    np.std(pseudo_preds) + 1e-8
                )
                signature_components.extend(
                    pseudo_preds[:10].tolist()
                )  # Take first 10 predictions

        return np.array(signature_components)

    def _compute_pairwise_mi(self, signatures: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise mutual information between model signatures.

        Args:
            signatures: List of model signatures

        Returns:
            MI matrix where mi_matrix[i][j] is MI between model i and model j
        """
        n_models = len(signatures)
        mi_matrix = np.zeros((n_models, n_models))

        # Ensure all signatures have the same length
        min_length = min(len(sig) for sig in signatures)
        signatures = [sig[:min_length] for sig in signatures]

        for i in range(n_models):
            for j in range(i, n_models):
                if i == j:
                    mi_matrix[i][j] = 1.0  # Perfect correlation with itself
                else:
                    # Compute MI between signature i and signature j
                    try:
                        # Discretize continuous values for MI computation
                        sig_i = self._discretize_signal(signatures[i])
                        sig_j = self._discretize_signal(signatures[j])

                        # Use mutual_info_regression as an approximation
                        # Note: This is a simplification for the simulation phase
                        mi_value = mutual_info_regression(
                            sig_i.reshape(-1, 1), sig_j, random_state=42
                        )[0]

                        # Normalize MI to [0, 1] range
                        mi_matrix[i][j] = mi_matrix[j][i] = min(1.0, max(0.0, mi_value))

                    except Exception as e:
                        print(
                            f"MIAggregationStrategy: Error computing MI between models {i} and {j}: {e}"
                        )
                        # Default to low similarity if computation fails
                        mi_matrix[i][j] = mi_matrix[j][i] = 0.0

        print(f"MIAggregationStrategy: Computed MI matrix with shape {mi_matrix.shape}")
        return mi_matrix

    def _discretize_signal(self, signal: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """
        Discretize a continuous signal for MI computation.

        Args:
            signal: Continuous signal to discretize
            n_bins: Number of bins for discretization

        Returns:
            Discretized signal
        """
        if len(signal) == 0:
            return signal

        # Handle edge case where all values are the same
        if np.std(signal) < 1e-8:
            return np.zeros_like(signal, dtype=int)

        # Use quantile-based binning for better distribution
        bins = np.quantile(signal, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates

        if len(bins) <= 1:
            return np.zeros_like(signal, dtype=int)

        discretized = np.digitize(signal, bins[1:-1])
        return discretized

    def _identify_trusted_models(self, mi_matrix: np.ndarray) -> List[int]:
        """
        Identify trusted models based on MI similarity patterns.

        Args:
            mi_matrix: Pairwise MI matrix

        Returns:
            List of indices of trusted models
        """
        n_models = mi_matrix.shape[0]

        # Compute average MI for each model with all others
        avg_mi_scores = []
        for i in range(n_models):
            # Exclude self-similarity (diagonal)
            other_mi = [mi_matrix[i][j] for j in range(n_models) if i != j]
            avg_mi = np.mean(other_mi) if other_mi else 0.0
            avg_mi_scores.append(avg_mi)

        # Identify outliers: models with significantly different MI patterns
        avg_mi_scores = np.array(avg_mi_scores)
        mi_mean = np.mean(avg_mi_scores)
        mi_std = np.std(avg_mi_scores)

        # Models with MI scores too far from the mean are considered suspicious
        trusted_indices = []
        for i, score in enumerate(avg_mi_scores):
            # Use a simple threshold: within 2 standard deviations of mean
            if mi_std > 0 and abs(score - mi_mean) <= 2 * mi_std:
                trusted_indices.append(i)
            elif mi_std == 0:  # All models have same MI (edge case)
                trusted_indices.append(i)

        # Ensure we have at least some trusted models
        if len(trusted_indices) == 0:
            print(
                "MIAggregationStrategy: Warning - No models passed MI filter, using model with highest average MI"
            )
            best_model = int(np.argmax(avg_mi_scores))
            trusted_indices = [best_model]

        print(
            f"MIAggregationStrategy: Identified {len(trusted_indices)} trusted models out of {n_models}"
        )
        print(f"MIAggregationStrategy: Average MI scores: {avg_mi_scores}")
        print(f"MIAggregationStrategy: Trusted model indices: {trusted_indices}")

        return trusted_indices

    def _weighted_average(
        self,
        model_updates: List[ModelWeights],
        data_sizes: Optional[List[int]] = None,
        previous_weights: Optional[ModelWeights] = None,
    ) -> ModelWeights:
        """
        Compute weighted average of trusted model updates.

        Args:
            model_updates: List of trusted model updates
            data_sizes: List of data sizes for weighting
            previous_weights: Previous global weights (for incremental updates)

        Returns:
            Aggregated model weights
        """
        if not model_updates:
            return previous_weights

        if len(model_updates) == 1:
            return model_updates[0]

        # Calculate weights for averaging
        if data_sizes is not None:
            total_data = sum(data_sizes)
            weights = [size / total_data for size in data_sizes]
        else:
            # Equal weighting if no data sizes provided
            weights = [1.0 / len(model_updates)] * len(model_updates)

        print(
            f"MIAggregationStrategy: Computing weighted average with weights: {weights}"
        )

        # Perform weighted aggregation
        aggregated_weights = {}
        first_update = model_updates[0]

        if isinstance(first_update, dict):
            # Dictionary format (typical for PyTorch state_dict)
            for param_name in first_update.keys():
                weighted_param = None

                for i, update in enumerate(model_updates):
                    param = update[param_name]
                    weight_scalar = weights[i]

                    # Ensure scalar is compatible with tensor type
                    if hasattr(param, "detach"):  # PyTorch tensor
                        import torch

                        if not isinstance(weight_scalar, torch.Tensor):
                            weight_scalar = torch.tensor(
                                weight_scalar, dtype=param.dtype, device=param.device
                            )

                    if weighted_param is None:
                        weighted_param = weight_scalar * param
                    else:
                        weighted_param += weight_scalar * param

                aggregated_weights[param_name] = weighted_param
        else:
            # Handle other formats (lists, single tensors, etc.)
            weighted_sum = None
            for i, update in enumerate(model_updates):
                weight_scalar = weights[i]

                # Ensure scalar is compatible with tensor type
                if hasattr(update, "detach"):  # PyTorch tensor
                    import torch

                    if not isinstance(weight_scalar, torch.Tensor):
                        weight_scalar = torch.tensor(
                            weight_scalar, dtype=update.dtype, device=update.device
                        )

                if weighted_sum is None:
                    weighted_sum = weight_scalar * update
                else:
                    weighted_sum += weight_scalar * update
            aggregated_weights = weighted_sum

        print("MIAggregationStrategy: Weighted aggregation completed")
        return aggregated_weights
