from flare.blockchain import MockChainConnector, MockPoAConsensus
from flare.compression import ZlibCompressor
from flare.core import FlareConfig
from flare.federation import FedAvg
from flare.models import MockModelAdapter
from flare.storage import InMemoryStorageProvider

config = FlareConfig()
config.set('model', MockModelAdapter())
config.set('compressor', ZlibCompressor())
config.set('blockchain_connector', MockChainConnector())
config.set('consensus_mechanism', MockPoAConsensus(config))
config.set('storage_provider', InMemoryStorageProvider())
config.set('aggregation_strategy', FedAvg())
config.set('num_rounds', 10)
config.set('clients_per_round', 5)


print("Flare components initialized successfully.")
print(config)
