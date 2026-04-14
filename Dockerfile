FROM nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Build-time arguments for user id mapping
ARG USERNAME
ARG UID
ARG GID

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    curl \
    git \
    python3.12-dev \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install packages there
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="${VENV_PATH}/bin:$PATH"
# activate the virtual environment
RUN . $VENV_PATH/bin/activate

# Create a user and group with the specified UID and GID
RUN groupadd --gid $GID $USERNAME && \
    useradd --no-log-init --uid $UID --gid $GID --create-home --shell /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install conch (git-based dependency not resolvable from pyproject.toml by pip)
RUN pip3 install --no-cache-dir git+https://github.com/Mahmoodlab/CONCH.git

# Copy and install plismbench (pulls in all remaining dependencies)
COPY --chown=${USERNAME}:${USERNAME} . /workspaces/plism-benchmark
RUN pip3 install --no-cache-dir -e /workspaces/plism-benchmark

# Install core PyTorch packages (CUDA-specific builds require explicit index)
# Must come after plismbench install to override the CPU torch pulled from PyPI
RUN pip3 install --no-cache-dir \
    torch==2.10.0+cu129 \
    torchvision==0.25.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# Clean pip3 cache
RUN rm -rf /root/.cache/pip

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# Switch to the new user
USER $USERNAME

# Install claude code 
RUN curl -fsSL https://claude.ai/install.sh | bash

# Set the working directory
WORKDIR /workspaces

# === ENVIRONMENT VARIABLES ===
ENV DEV_CWD=/workspaces/plism-benchmark

ENV HUGGINGFACE_HUB_CACHE=/mnt/.cache
ENV TORCH_HOME=/mnt/.torchhome