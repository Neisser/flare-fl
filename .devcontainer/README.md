# Flare FL Dev Container

Este directorio contiene la configuración para desarrollar Flare usando VS Code/Cursor Dev Containers.

## 🚀 Características

- **Python 3.12** con todas las herramientas de desarrollo
- **Soporte GPU/CUDA** automático (si está disponible)
- **Linting y formateo** configurado (flake8, pylint, black, isort)
- **Testing** con pytest
- **Type checking** con mypy
- **Jupyter** para notebooks
- **Git** integrado
- **Pre-commit hooks** para calidad de código

## 📋 Requisitos

- VS Code o Cursor con extensión "Dev Containers"
- Docker Desktop con soporte GPU (opcional, para CUDA)

## 🔧 Uso

1. **Abrir en Dev Container:**
   - Ctrl+Shift+P → "Dev Containers: Reopen in Container"
   - O usar el botón "Reopen in Container" en la esquina inferior izquierda

2. **Primera vez:**
   - El contenedor se construirá automáticamente
   - Se instalarán todas las dependencias de Flare
   - Se configurará el entorno de desarrollo

3. **Desarrollo:**
   - El linter usará automáticamente el Python del contenedor
   - Todas las extensiones están pre-configuradas
   - Formateo automático al guardar

## 🐳 Configuración Docker

### GPU Support
Si tienes GPU NVIDIA:
```bash
# Verificar GPU
nvidia-smi

# El contenedor detectará automáticamente CUDA
# y instalará PyTorch con soporte GPU
```

### Sin GPU
El contenedor funcionará con PyTorch CPU-only.

## 🧪 Testing

```bash
# Ejecutar tests
pytest

# Con coverage
pytest --cov=flare

# Tests específicos
pytest tests/core/
```

## 📝 Linting y Formateo

```bash
# Formatear código
black .
isort .

# Linting
flake8 .
pylint flare/
mypy flare/
```

## 🔄 Actualizar Contenedor

Si cambias el Dockerfile:
1. Ctrl+Shift+P → "Dev Containers: Rebuild Container"
2. O eliminar el contenedor y recrearlo

## 🚨 Troubleshooting

### Error de permisos
```bash
# Dentro del contenedor
sudo chown -R developer:developer /app
```

### GPU no detectada
```bash
# Verificar drivers NVIDIA
nvidia-smi

# Verificar Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Dependencias faltantes
```bash
# Reinstalar
pip install -e .
pip install -r requirements-dev.txt
```

## 📚 Comandos Útiles

```bash
# Verificar Python y PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"

# Verificar GPU
nvidia-smi

# Verificar instalación Flare
python -c "import flare; print('Flare importado correctamente')"

# Ejecutar ejemplo
python -m examples.builder_example
```
