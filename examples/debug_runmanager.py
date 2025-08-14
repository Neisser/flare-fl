"""
Script de debug para probar RunManager aisladamente.
"""

import json

from flare.simulation.run_manager import RunManager, RunMetadata


def test_runmanager():
    print("🔍 Testing RunManager...")

    # 1. Crear RunManager
    rm = RunManager("debug_runs")
    print("✅ RunManager creado")

    # 2. Crear metadata directamente
    metadata = RunMetadata(scenario_name="debug_test", description="Test debug")
    print(f"✅ Metadata creado: {metadata.run_id}")

    # 3. Intentar guardar metadata
    try:
        rm._save_metadata(metadata)
        print("✅ Metadata guardado")

        # Verificar que el archivo se creó
        metadata_path = rm.base_path / "metadata" / f"{metadata.run_id}.json"
        if metadata_path.exists():
            print(f"✅ Archivo existe: {metadata_path}")
            with open(metadata_path, "r") as f:
                content = f.read()
                print(f"📄 Tamaño: {len(content)} bytes")
                if content:
                    print(f"📄 Contenido: {content[:200]}...")
                else:
                    print("❌ Archivo está vacío!")
        else:
            print("❌ Archivo no existe!")

    except Exception as e:
        print(f"❌ Error guardando: {e}")
        import traceback

        traceback.print_exc()

    # 4. Intentar cargar metadata
    try:
        loaded = rm._load_metadata(metadata.run_id)
        print(f"✅ Metadata cargado: {loaded.scenario_name}")
    except Exception as e:
        print(f"❌ Error cargando: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_runmanager()
