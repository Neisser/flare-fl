from typing import Dict, Optional

from .providers import StorageData, StorageIdentifier, StorageProvider


class InMemoryStorageProvider(StorageProvider):
    """Stores data in an in-memory dictionary. Useful for simulations."""
    def __init__(self):
        self._store: Dict[StorageIdentifier, StorageData] = {}
        print("InMemoryStorageProvider initialized.")

    def put(
            self,
            identifier: StorageIdentifier,
            data: StorageData
            ) -> Optional[StorageIdentifier]:
        print(f"InMemoryStorage: put '{identifier}' ({len(data)} bytes).")
        self._store[identifier] = data
        return identifier

    def get(self, identifier: StorageIdentifier) -> Optional[StorageData]:
        print(f"InMemoryStorage: get '{identifier}'.")
        data = self._store.get(identifier)
        if data:
            print(f"InMemoryStorage: found '{identifier}' ({len(data)} bytes).")
        else:
            print(f"InMemoryStorage: '{identifier}' not found.")
        return data

    def delete(self, identifier: StorageIdentifier) -> bool:
        print(f"InMemoryStorage: delete '{identifier}'.")
        if identifier in self._store:
            del self._store[identifier]
            print(f"InMemoryStorage: '{identifier}' deleted.")
            return True
        print(f"InMemoryStorage: '{identifier}' not found for deletion.")
        return False

    def exists(self, identifier: StorageIdentifier) -> bool:
        exists = identifier in self._store
        print(f"InMemoryStorage: exists '{identifier}'? {exists}.")
        return exists
