from abc import ABC, abstractmethod

# For compressed and non-compressed data
BytesLike = bytes


class Compressor(ABC):
    """Abstract base class for data compressors."""

    @abstractmethod
    def compress(self, data: BytesLike) -> BytesLike:
        """Compresses the input data."""
        pass

    @abstractmethod
    def decompress(self, data: BytesLike) -> BytesLike:
        """Decompresses the input data."""
        pass
