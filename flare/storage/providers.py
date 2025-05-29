from abc import ABC, abstractmethod
from typing import Optional

StorageIdentifier = str

# Assume we store bytes
StorageData = bytes


class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    def put(
        self,
        identifier: StorageIdentifier,
        data: StorageData
    ) -> Optional[StorageIdentifier]:
        """
        Stores data and returns an identifier (e.g., path, CID, key).
        May return the same identifier if it's deterministic or None if not applicable.
        """
        pass

    @abstractmethod
    def get(self, identifier: StorageIdentifier) -> Optional[StorageData]:
        """Retrieves data by its identifier."""
        pass

    @abstractmethod
    def delete(self, identifier: StorageIdentifier) -> bool:
        """Deletes data by its identifier. Returns True on success."""
        pass

    @abstractmethod
    def exists(self, identifier: StorageIdentifier) -> bool:
        """Checks if data with the identifier exists."""
        pass
