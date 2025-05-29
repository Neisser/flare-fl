import pickle
from typing import Tuple

import numpy as np

from .compressors import BytesLike, Compressor


class PowerSGDCompressor(Compressor):
    """
    PowerSGD (Power Iteration-based Stochastic Gradient Descent) Compressor.

    Implements low-rank compression of gradients/weight differences using power iteration
    to find the dominant singular vectors, as described in BEFL Algorithm 2.

    The algorithm compresses matrices by approximating them as a product of two smaller matrices:
    M ≈ P @ Q^T, where P and Q have significantly fewer columns than M.
    """

    def __init__(
        self,
        rank: int = 4,
        power_iterations: int = 1,
        min_compression_rate: float = 2.0,
    ):
        """
        Initialize PowerSGD compressor.

        Args:
            rank: Target rank for low-rank approximation (r in the algorithm)
            power_iterations: Number of power iterations for SVD approximation
            min_compression_rate: Minimum compression rate to apply PowerSGD (otherwise fallback)
        """
        self.rank = rank
        self.power_iterations = power_iterations
        self.min_compression_rate = min_compression_rate
        print(
            f"PowerSGDCompressor initialized with rank={rank}, power_iterations={power_iterations}"
        )

    def compress(self, data: BytesLike) -> BytesLike:
        """
        Compresses serialized model weights using PowerSGD low-rank approximation.

        Algorithm:
        1. Deserialize weights from bytes
        2. For each weight matrix M with shape (m, n):
           - If compression is beneficial: compute M ≈ P @ Q^T where P: (m, r), Q: (n, r)
           - Use power iteration to find dominant r singular vectors
           - Store P and Q instead of full matrix M
        3. Serialize compressed representation
        """
        print(f"PowerSGDCompressor: compressing {len(data)} bytes...")

        try:
            # Deserialize the model weights
            weights = pickle.loads(data)
            compressed_weights = {}

            total_original_params = 0
            total_compressed_params = 0

            # Process each weight tensor/matrix
            for name, weight in (
                weights.items() if isinstance(weights, dict) else enumerate(weights)
            ):
                # Convert PyTorch tensor to numpy array for processing
                if hasattr(weight, "detach"):
                    # PyTorch tensor
                    weight_array = weight.detach().cpu().numpy()
                else:
                    # Assume it's already a numpy array
                    weight_array = np.array(weight)

                if weight_array.ndim >= 2:
                    # Only compress 2D+ tensors (matrices)
                    original_shape = weight_array.shape
                    original_params = weight_array.size  # Use .size not .size()
                    total_original_params += original_params

                    # Reshape to 2D matrix for compression
                    if weight_array.ndim > 2:
                        # Flatten all but the last dimension
                        matrix = weight_array.reshape(-1, weight_array.shape[-1])
                    else:
                        matrix = weight_array

                    m, n = matrix.shape

                    # Check if compression is beneficial
                    compressed_params = (m + n) * self.rank
                    compression_rate = (
                        original_params / compressed_params
                        if compressed_params > 0
                        else 1
                    )

                    if (
                        compression_rate >= self.min_compression_rate
                        and min(m, n) > self.rank
                    ):
                        # Apply PowerSGD compression
                        P, Q = self._power_sgd_compress(matrix)
                        compressed_weights[name] = {
                            "type": "powersgd",
                            "P": P,
                            "Q": Q,
                            "original_shape": original_shape,
                            "rank": self.rank,
                        }
                        total_compressed_params += P.size + Q.size
                        print(
                            f"  {name}: {original_shape} -> rank-{self.rank} (rate: {compression_rate:.2f}x)"
                        )
                    else:
                        # Keep original if compression not beneficial
                        compressed_weights[name] = {
                            "type": "original",
                            "data": weight_array,
                            "original_shape": original_shape,
                        }
                        total_compressed_params += original_params
                        print(
                            f"  {name}: {original_shape} -> kept original (low compression benefit)"
                        )
                else:
                    # Keep scalars, 1D tensors, or non-numeric data as-is
                    compressed_weights[name] = {
                        "type": "original",
                        "data": weight_array,
                        "original_shape": weight_array.shape
                        if hasattr(weight_array, "shape")
                        else None,
                    }
                    if hasattr(weight_array, "size"):
                        total_compressed_params += weight_array.size
                        total_original_params += weight_array.size

            # Serialize compressed representation
            compressed_data = pickle.dumps(compressed_weights)

            overall_compression_rate = (
                total_original_params / total_compressed_params
                if total_compressed_params > 0
                else 1
            )
            print(
                f"PowerSGDCompressor: compressed to {len(compressed_data)} bytes "
                f"(params: {total_original_params} -> {total_compressed_params}, "
                f"rate: {overall_compression_rate:.2f}x)"
            )

            return compressed_data

        except Exception as e:
            print(f"PowerSGDCompressor: Error during compression: {e}")
            # Fallback: return original data
            return data

    def decompress(self, data: BytesLike) -> BytesLike:
        """
        Decompresses PowerSGD-compressed model weights.
        """
        print(f"PowerSGDCompressor: decompressing {len(data)} bytes...")

        try:
            # Deserialize compressed representation
            compressed_weights = pickle.loads(data)
            reconstructed_weights = {}

            # Reconstruct each weight tensor
            for name, weight_info in (
                compressed_weights.items()
                if isinstance(compressed_weights, dict)
                else enumerate(compressed_weights)
            ):
                if weight_info["type"] == "powersgd":
                    # Reconstruct from PowerSGD compression
                    P = weight_info["P"]
                    Q = weight_info["Q"]
                    original_shape = weight_info["original_shape"]

                    # Reconstruct matrix: M ≈ P @ Q^T
                    reconstructed_matrix = P @ Q.T

                    # Reshape back to original dimensions
                    reconstructed_weight = reconstructed_matrix.reshape(original_shape)
                    reconstructed_weights[name] = reconstructed_weight

                elif weight_info["type"] == "original":
                    # Use original uncompressed data
                    reconstructed_weights[name] = weight_info["data"]
                else:
                    print(
                        f"PowerSGDCompressor: Unknown compression type: {weight_info['type']}"
                    )
                    reconstructed_weights[name] = weight_info["data"]

            # Serialize reconstructed weights
            decompressed_data = pickle.dumps(reconstructed_weights)
            print(
                f"PowerSGDCompressor: decompressed to {len(decompressed_data)} bytes."
            )

            return decompressed_data

        except Exception as e:
            print(f"PowerSGDCompressor: Error during decompression: {e}")
            # Fallback: return original data
            return data

    def _power_sgd_compress(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies PowerSGD compression to a single matrix using power iteration.

        Algorithm (BEFL Algorithm 2):
        1. Initialize random matrix Q with r columns
        2. For t iterations:
           - P = M @ Q
           - Orthogonalize P (QR decomposition)
           - Q = M^T @ P
           - Orthogonalize Q (QR decomposition)
        3. Return P and Q such that M ≈ P @ Q^T

        Args:
            matrix: Input matrix of shape (m, n)

        Returns:
            P: Left factor matrix of shape (m, rank)
            Q: Right factor matrix of shape (n, rank)
        """
        m, n = matrix.shape

        # Initialize random orthogonal matrix Q
        np.random.seed(42)  # For reproducible results in simulation
        Q = np.random.randn(n, self.rank)
        Q, _ = np.linalg.qr(Q)  # Orthogonalize

        # Power iteration to find dominant subspace
        for iteration in range(self.power_iterations):
            # P = M @ Q
            P = matrix @ Q

            # Orthogonalize P using QR decomposition
            P, _ = np.linalg.qr(P)

            # Q = M^T @ P
            Q = matrix.T @ P

            # Orthogonalize Q using QR decomposition
            Q, _ = np.linalg.qr(Q)

        # Final computation: P = M @ Q
        P = matrix @ Q

        return P, Q

    def compute_compression_error(self, original_data: BytesLike) -> float:
        """
        Utility method to compute the L2 error between original and compressed-decompressed data.
        Useful for testing compression quality.

        Returns:
            L2 reconstruction error (||original - reconstructed||_2)
        """
        try:
            compressed_data = self.compress(original_data)
            reconstructed_data = self.decompress(compressed_data)

            original_weights = pickle.loads(original_data)
            reconstructed_weights = pickle.loads(reconstructed_data)

            total_error = 0.0
            total_norm = 0.0

            for name in original_weights:
                # Convert to numpy arrays for comparison
                if hasattr(original_weights[name], "detach"):
                    orig = original_weights[name].detach().cpu().numpy()
                else:
                    orig = np.array(original_weights[name])

                if hasattr(reconstructed_weights[name], "detach"):
                    recon = reconstructed_weights[name].detach().cpu().numpy()
                else:
                    recon = np.array(reconstructed_weights[name])

                error = np.linalg.norm(orig - recon)
                norm = np.linalg.norm(orig)

                total_error += error**2
                total_norm += norm**2

            relative_error = (
                np.sqrt(total_error) / np.sqrt(total_norm) if total_norm > 0 else 0.0
            )
            return relative_error

        except Exception as e:
            print(f"PowerSGDCompressor: Error computing compression error: {e}")
            return float("inf")
