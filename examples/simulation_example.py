"""
Ejemplo de uso del módulo de simulación de Flare.

Este ejemplo demuestra cómo:
1. Crear un escenario de simulación
2. Ejecutar la simulación
3. Recolectar métricas
4. Gestionar corridas
5. Visualizar resultados
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from flare.builder import ClientBuilder, OrchestratorBuilder
from flare.compression import PowerSGDCompressor
from flare.federation import FedAvg
from flare.models import PyTorchModelAdapter
from flare.simulation import (
    MetricsCollector,
    MetricType,
    RunManager,
    RunMetadata,
    Scenario,
    ScenarioBuilder,
    SimulationVisualizer,
)


# 1. Definir modelo simple para MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


def main():
    # 2. Preparar datos
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 3. Crear modelo y adaptador
    model = SimpleCNN()
    model_adapter = PyTorchModelAdapter(model)

    # 4. Configurar escenario usando el Builder
    scenario = (
        ScenarioBuilder(
            name="mnist_fedavg_powersgd",
            description="MNIST training with FedAvg and PowerSGD compression",
        )
        .with_orchestrator(
            OrchestratorBuilder()
            .with_model_adapter(model_adapter)
            .with_compressor(PowerSGDCompressor(rank=4))
            .with_aggregation_strategy(FedAvg())
        )
        .with_clients(
            [
                ClientBuilder()
                .with_model_adapter(model_adapter)
                .with_compressor(PowerSGDCompressor(rank=4))
                .with_optimizer(optim.Adam(model.parameters(), lr=0.01))
                .with_criterion(nn.CrossEntropyLoss())
                .with_data_loader(
                    DataLoader(train_dataset, batch_size=32, shuffle=True)
                )
                for _ in range(5)  # 5 clientes
            ]
        )
        .with_rounds(10)
        .with_batch_size(32)
        .with_learning_rate(0.01)
        .with_tags(["mnist", "fedavg", "powersgd", "example"])
        .build()
    )

    # 5. Inicializar gestor de corridas y visualizador
    run_manager = RunManager("runs")
    visualizer = SimulationVisualizer(run_manager)

    # 6. Crear nueva corrida
    run_metadata = run_manager.create_run(
        scenario=scenario,
        description="MNIST training with FedAvg and PowerSGD compression",
        tags=["mnist", "fedavg", "powersgd", "example"],
    )

    # 7. Inicializar recolector de métricas
    metrics = MetricsCollector(scenario.name)
    metrics.register_metric(MetricType.LOSS, "training_loss")
    metrics.register_metric(MetricType.ACCURACY, "training_accuracy")
    metrics.register_metric(MetricType.COMMUNICATION_BYTES, "communication")

    # 8. Ejecutar simulación
    try:
        run_manager.update_run_status(run_metadata.run_id, "running")

        for round_num in range(scenario.num_rounds):
            # Entrenamiento local en clientes
            client_metrics = []
            for client in scenario.clients:
                loss, accuracy = client.train_round()
                client_metrics.append({"loss": loss, "accuracy": accuracy})

            # Agregación en orquestador
            aggregated_model = scenario.orchestrator.aggregate_round()

            # Recolectar métricas
            avg_loss = sum(m["loss"] for m in client_metrics) / len(client_metrics)
            avg_accuracy = sum(m["accuracy"] for m in client_metrics) / len(
                client_metrics
            )

            metrics.record_value("training_loss", avg_loss, {"round": round_num})
            metrics.record_value(
                "training_accuracy", avg_accuracy, {"round": round_num}
            )
            metrics.record_value(
                "communication",
                scenario.orchestrator.get_communication_bytes(),
                {"round": round_num},
            )

            # Guardar modelo
            run_manager.save_model(run_metadata.run_id, "model.pt", round_num=round_num)

        run_manager.update_run_status(run_metadata.run_id, "completed")

    except Exception as e:
        run_manager.update_run_status(
            run_metadata.run_id, "failed", error_message=str(e)
        )
        raise

    finally:
        # 9. Guardar métricas y logs
        run_manager.save_metrics(run_metadata.run_id, metrics)
        run_manager.save_logs(
            run_metadata.run_id,
            {"scenario": scenario.to_dict(), "client_metrics": client_metrics},
        )

    # 10. Generar visualizaciones
    visualizer.plot_metric_evolution(
        run_metadata.run_id,
        "training_loss",
        save_path=f"runs/{run_metadata.run_id}/plots/loss_evolution.png",
    )

    visualizer.plot_metric_evolution(
        run_metadata.run_id,
        "training_accuracy",
        save_path=f"runs/{run_metadata.run_id}/plots/accuracy_evolution.png",
    )

    # 11. Generar reporte completo
    visualizer.generate_run_report(
        run_metadata.run_id, f"runs/{run_metadata.run_id}/report"
    )


if __name__ == "__main__":
    main()
