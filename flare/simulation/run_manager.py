"""
Run Manager Module for Flare Simulations

Gestiona el almacenamiento, versionado y recuperación de corridas de simulación.
"""

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .metrics import MetricsCollector
from .scenario import Scenario


@dataclass
class RunMetadata:
    """Metadata para una corrida de simulación."""

    scenario_name: str
    description: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"  # created, running, completed, failed
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la metadata a un diccionario."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "scenario_name": self.scenario_name,
            "description": self.description,
            "tags": self.tags,
            "parameters": self.parameters,
            "status": self.status,
            "error_message": self.error_message,
        }


class RunManager:
    """Gestiona el almacenamiento y versionado de corridas de simulación."""

    def __init__(self, base_path: Union[str, Path] = "runs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Crear estructura de directorios
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "metrics").mkdir(exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        (self.base_path / "logs").mkdir(exist_ok=True)

    def create_run(
        self,
        scenario: Scenario,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> RunMetadata:
        """Crea una nueva corrida para un escenario."""
        metadata = RunMetadata(
            scenario_name=scenario.name,
            description=description or scenario.description,
            tags=tags or scenario.tags,
            parameters=scenario.parameters,
        )

        # Crear directorio para la corrida
        run_path = self.base_path / metadata.run_id
        run_path.mkdir(parents=True)

        # Guardar metadata inicial
        self._save_metadata(metadata)

        # Guardar configuración del escenario
        # TODO: Fix corruption issue with scenario.save_metadata()
        # scenario.save_metadata(run_path)

        return metadata

    def update_run_status(
        self, run_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        """Actualiza el estado de una corrida."""
        metadata = self._load_metadata(run_id)
        metadata.status = status
        metadata.error_message = error_message
        self._save_metadata(metadata)

    def save_metrics(self, run_id: str, metrics: MetricsCollector) -> None:
        """Guarda las métricas de una corrida."""
        run_path = self.base_path / run_id
        metrics.save_metrics(run_path)

    def save_model(
        self, run_id: str, model_path: Union[str, Path], round_num: Optional[int] = None
    ) -> None:
        """Guarda un modelo de una corrida."""
        run_path = self.base_path / run_id / "models"
        if round_num is not None:
            run_path = run_path / f"round_{round_num}"
        run_path.mkdir(parents=True, exist_ok=True)

        # Copiar el modelo al directorio de la corrida
        shutil.copy2(model_path, run_path / "model.pt")

    def save_logs(self, run_id: str, logs: Dict[str, Any]) -> None:
        """Guarda logs de una corrida."""
        run_path = self.base_path / run_id
        with open(run_path / "logs.json", "w") as f:
            json.dump(logs, f, indent=2)

    def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Recupera la metadata de una corrida."""
        try:
            return self._load_metadata(run_id)
        except FileNotFoundError:
            return None

    def list_runs(
        self,
        scenario_name: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[RunMetadata]:
        """Lista todas las corridas que coinciden con los filtros."""
        runs = []
        for metadata_file in (self.base_path / "metadata").glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)

                    # Convertir timestamp de string a datetime
                    if isinstance(data.get("timestamp"), str):
                        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                    metadata = RunMetadata(**data)

                    # Aplicar filtros
                    if scenario_name and metadata.scenario_name != scenario_name:
                        continue
                    if status and metadata.status != status:
                        continue
                    if tags and not all(tag in metadata.tags for tag in tags):
                        continue

                    runs.append(metadata)
            except Exception as e:
                print(f"Error loading {metadata_file}: {e}")

        return sorted(runs, key=lambda x: x.timestamp, reverse=True)

    def delete_run(self, run_id: str) -> None:
        """Elimina una corrida y todos sus datos."""
        run_path = self.base_path / run_id
        if run_path.exists():
            shutil.rmtree(run_path)

        metadata_path = self.base_path / "metadata" / f"{run_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()

    def _save_metadata(self, metadata: RunMetadata) -> None:
        """Guarda la metadata de una corrida."""
        metadata_path = self.base_path / "metadata" / f"{metadata.run_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _load_metadata(self, run_id: str) -> RunMetadata:
        """Carga la metadata de una corrida."""
        metadata_path = self.base_path / "metadata" / f"{run_id}.json"
        with open(metadata_path, "r") as f:
            data = json.load(f)

            # Convertir timestamp de string a datetime
            if isinstance(data.get("timestamp"), str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])

            return RunMetadata(**data)

    def export_run(self, run_id: str, export_path: Union[str, Path]) -> None:
        """Exporta una corrida completa a un directorio."""
        run_path = self.base_path / run_id
        export_path = Path(export_path)

        if not run_path.exists():
            raise ValueError(f"Run {run_id} not found")

        # Crear directorio de exportación
        export_path.mkdir(parents=True, exist_ok=True)

        # Copiar todos los archivos
        shutil.copytree(run_path, export_path / run_id, dirs_exist_ok=True)

        # Copiar metadata
        shutil.copy2(
            self.base_path / "metadata" / f"{run_id}.json",
            export_path / "metadata.json",
        )

        # Crear archivo de resumen
        metadata = self._load_metadata(run_id)
        summary = {
            "run_id": metadata.run_id,
            "scenario_name": metadata.scenario_name,
            "timestamp": metadata.timestamp.isoformat(),
            "status": metadata.status,
            "description": metadata.description,
            "tags": metadata.tags,
            "parameters": metadata.parameters,
        }

        with open(export_path / "summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
