"""
Scenario Module for Flare Simulations

Define la estructura y configuración de escenarios de simulación para Federated Learning.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from ..blockchain import BlockchainConnector
from ..builder import ClientBuilder, OrchestratorBuilder
from ..compression import Compressor
from ..consensus import ConsensusMechanism
from ..federation import Client, Orchestrator
from ..federation.aggregation_strategies import AggregationStrategy
from ..models import ModelAdapter
from ..storage import StorageProvider


@dataclass
class Scenario:
    """Representa un escenario completo de simulación FL."""

    name: str
    description: str

    # Componentes principales
    orchestrator: Optional[Orchestrator]
    clients: List[Client]

    # Configuración
    num_rounds: int
    batch_size: int
    learning_rate: float

    timestamp: datetime = field(default_factory=datetime.now)

    # Componentes opcionales
    model_adapter: Optional[ModelAdapter] = None
    compressor: Optional[Compressor] = None
    aggregation_strategy: Optional[AggregationStrategy] = None
    consensus_mechanism: Optional[ConsensusMechanism] = None
    blockchain_connector: Optional[BlockchainConnector] = None
    storage_provider: Optional[StorageProvider] = None

    # Metadata adicional
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el escenario a un diccionario para serialización."""
        return {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "num_rounds": self.num_rounds,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "tags": self.tags,
            "parameters": self.parameters,
            "orchestrator_type": type(self.orchestrator).__name__
            if self.orchestrator
            else None,
            "num_clients": len(self.clients),
            "client_types": [type(c).__name__ for c in self.clients],
            "has_model_adapter": self.model_adapter is not None,
            "has_compressor": self.compressor is not None,
            "has_aggregation": self.aggregation_strategy is not None,
            "has_consensus": self.consensus_mechanism is not None,
            "has_blockchain": self.blockchain_connector is not None,
            "has_storage": self.storage_provider is not None,
        }

    def save_metadata(self, path: Union[str, Path]) -> None:
        """Guarda los metadatos del escenario en un archivo JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path / "scenario_metadata.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ScenarioBuilder:
    """Builder para crear escenarios de simulación de manera fluida."""

    def __init__(self, name: str, description: str):
        self._scenario = Scenario(
            name=name,
            description=description,
            orchestrator=cast(Optional[Orchestrator], None),  # Se establecerá después
            clients=[],
            num_rounds=10,  # Defaults razonables
            batch_size=32,
            learning_rate=0.01,
        )

    def with_orchestrator(self, builder: OrchestratorBuilder) -> "ScenarioBuilder":
        """Configura el orquestador usando un OrchestratorBuilder."""
        self._scenario.orchestrator = builder.build()
        return self

    def with_clients(self, builders: List[ClientBuilder]) -> "ScenarioBuilder":
        """Configura los clientes usando una lista de ClientBuilder."""
        self._scenario.clients = [b.build() for b in builders]
        return self

    def with_rounds(self, num_rounds: int) -> "ScenarioBuilder":
        """Establece el número de rondas de entrenamiento."""
        self._scenario.num_rounds = num_rounds
        return self

    def with_batch_size(self, batch_size: int) -> "ScenarioBuilder":
        """Establece el tamaño de batch para entrenamiento."""
        self._scenario.batch_size = batch_size
        return self

    def with_learning_rate(self, lr: float) -> "ScenarioBuilder":
        """Establece la tasa de aprendizaje."""
        self._scenario.learning_rate = lr
        return self

    def with_tags(self, tags: List[str]) -> "ScenarioBuilder":
        """Añade tags al escenario."""
        self._scenario.tags = tags
        return self

    def with_parameters(self, parameters: Dict[str, Any]) -> "ScenarioBuilder":
        """Añade parámetros adicionales al escenario."""
        self._scenario.parameters = parameters
        return self

    def build(self) -> Scenario:
        """Construye y valida el escenario."""
        if not self._scenario.orchestrator:
            raise ValueError("Orchestrator must be configured before building scenario")
        if not self._scenario.clients:
            raise ValueError("At least one client must be configured")

        return self._scenario
