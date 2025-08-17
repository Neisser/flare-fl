FROM mcr.microsoft.com/devcontainers/python:3.12

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    tree \
    unzip \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development tools
RUN pip install --no-cache-dir \
    flake8 \
    pylint \
    black \
    isort \
    mypy \
    pytest \
    pytest-cov \
    ipython \
    jupyter \
    ipykernel

# Install PyTorch (will auto-detect CUDA if available)
RUN pip install --no-cache-dir torch torchvision torchaudio

# Create workspace directory
WORKDIR /workspaces/flare-fl

# Set up Python path
ENV PYTHONPATH=/workspaces/flare-fl

# Clean up any existing egg-info directories and ensure proper permissions
RUN mkdir -p /workspaces/flare-fl && \
    chown -R vscode:vscode /workspaces

# Install git hooks and pre-commit
RUN pip install --user pre-commit

# Add user's local bin to PATH
ENV PATH="/home/vscode/.local/bin:$PATH"

# Expose port for Jupyter if needed
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
