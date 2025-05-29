from abc import ABC, abstractmethod
from typing import List, Optional

from flare.models import ModelWeights  # We use the type ModelWeights from flare.models


class AggregationStrategy(ABC):
    """Abstract base class for model aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        local_model_updates: List[ModelWeights],
        # For weighted averaging, we can use client data sizes
        client_data_sizes: Optional[List[int]] = None,
        # For strategies that need it, we can pass the previous global model weights
        previous_global_weights: Optional[ModelWeights] = None,
    ) -> ModelWeights:
        """
        Aggregates local model updates to produce a new global model's weights.
        - local_model_updates: A list of model weights (or weight diffs) from clients.
        - client_data_sizes: Optional list of dataset sizes for weighted averaging.
        - previous_global_weights: Optional, weights of the global model before this round.
        """
        pass
