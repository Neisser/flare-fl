"""
Ejemplo simplificado de simulación de Flare.

Este ejemplo demuestra el sistema de simulación sin usar PyTorch
para evitar dependencias complejas durante la prueba inicial.
"""

import random

from flare.simulation import MetricsCollector, MetricType, RunManager, Scenario


# Mock de componentes para el ejemplo
class MockOrchestrator:
    def __init__(self):
        self.rounds_completed = 0

    def get_communication_bytes(self):
        return random.randint(1000, 5000)


class MockClient:
    def __init__(self, client_id):
        self.client_id = client_id

    def train_round(self):
        # Simular entrenamiento con métricas aleatorias
        loss = random.uniform(0.1, 2.0)
        accuracy = random.uniform(0.7, 0.95)
        return loss, accuracy


def main():
    print("🚀 Iniciando simulación de prueba del sistema de simulación...")

    # 1. Crear componentes mock
    orchestrator = MockOrchestrator()
    clients = [MockClient(f"client_{i}") for i in range(3)]

    # 2. Crear escenario manualmente (sin builders por ahora)
    scenario = Scenario(
        name="test_simulation",
        description="Simulación de prueba del sistema",
        orchestrator=orchestrator,
        clients=clients,
        num_rounds=5,
        batch_size=32,
        learning_rate=0.01,
        tags=["test", "simple", "mock"],
    )

    # 3. Inicializar gestor de corridas
    run_manager = RunManager("test_runs")

    # 4. Crear nueva corrida
    print("📝 Creando nueva corrida...")
    run_metadata = run_manager.create_run(
        scenario=scenario,
        description="Prueba del sistema de simulación",
        tags=["test", "mock"],
    )

    print(f"📋 Corrida creada: {run_metadata.run_id}")

    # Debug: verificar que el archivo de metadata se creó
    metadata_path = run_manager.base_path / "metadata" / f"{run_metadata.run_id}.json"
    print(f"🔍 Verificando metadata en: {metadata_path}")
    if metadata_path.exists():
        print("✅ Archivo de metadata existe")
        with open(metadata_path, "r") as f:
            content = f.read()
            print(f"📄 Contenido: {content[:100]}...")
    else:
        print("❌ Archivo de metadata NO existe")
        return None

    # 5. Inicializar recolector de métricas
    metrics = MetricsCollector(scenario.name)
    metrics.register_metric(MetricType.LOSS, "training_loss")
    metrics.register_metric(MetricType.ACCURACY, "training_accuracy")
    metrics.register_metric(MetricType.COMMUNICATION_BYTES, "communication")

    # 6. Ejecutar simulación
    try:
        run_manager.update_run_status(run_metadata.run_id, "running")
        print("▶️  Ejecutando simulación...")

        for round_num in range(scenario.num_rounds):
            print(f"  Ronda {round_num + 1}/{scenario.num_rounds}")

            # Entrenamiento local en clientes
            client_metrics = []
            for client in scenario.clients:
                loss, accuracy = client.train_round()
                client_metrics.append({"loss": loss, "accuracy": accuracy})
                print(f"{client.client_id}: loss={loss:.3f}, acc={accuracy:.3f}")

            # Recolectar métricas
            avg_loss = sum(m["loss"] for m in client_metrics) / len(client_metrics)
            avg_accuracy = sum(m["accuracy"] for m in client_metrics) / len(
                client_metrics
            )
            comm_bytes = orchestrator.get_communication_bytes()

            metrics.record_value("training_loss", avg_loss, {"round": round_num})
            metrics.record_value(
                "training_accuracy", avg_accuracy, {"round": round_num}
            )
            metrics.record_value("communication", comm_bytes, {"round": round_num})

            print(
                f"    📊 Promedio: loss={avg_loss:.3f}, acc={avg_accuracy:.3f}, comm={comm_bytes}B"
            )

        run_manager.update_run_status(run_metadata.run_id, "completed")
        print("✅ Simulación completada exitosamente")

    except Exception as e:
        run_manager.update_run_status(
            run_metadata.run_id, "failed", error_message=str(e)
        )
        print(f"❌ Error en la simulación: {e}")
        raise

    finally:
        # 7. Guardar métricas y logs
        run_manager.save_metrics(run_metadata.run_id, metrics)
        run_manager.save_logs(
            run_metadata.run_id,
            {
                "scenario": scenario.to_dict(),
                "total_clients": len(clients),
                "completed_rounds": scenario.num_rounds,
            },
        )
        print("💾 Métricas y logs guardados")

    # 8. Mostrar resumen
    print("\n📈 Resumen de la simulación:")
    print(f"  Run ID: {run_metadata.run_id}")
    print(f"  Escenario: {scenario.name}")
    print(f"  Rondas: {scenario.num_rounds}")
    print(f"  Clientes: {len(clients)}")
    print(f"  Estado: {run_manager._load_metadata(run_metadata.run_id).status}")
    print(f"  Directorio: test_runs/{run_metadata.run_id}")

    return run_metadata.run_id


if __name__ == "__main__":
    run_id = main()
    print(f"\n🎉 Simulación completada. Run ID: {run_id}")
