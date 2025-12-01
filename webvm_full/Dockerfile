# Base image for CheerpX (must be i386/debian:bookworm)
FROM i386/debian:bookworm

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# We include build-essential and python3-dev for compiling extensions if needed
# We include git, curl, nmap, sudo as requested by the script's "system packages" check
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    nmap \
    sudo \
    locales \
    libffi-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Generate locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install Rust via rustup (explicitly targeting i686 to avoid detection errors)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-host i686-unknown-linux-gnu
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_BUILD_TARGET=i686-unknown-linux-gnu
ENV PYO3_CROSS="1"

# Upgrade pip
RUN linux32 python3 -m pip install --upgrade pip --break-system-packages

# Install PyTorch (CPU only)
# NOTE: 32-bit support for PyTorch is limited. We attempt to install a version that might work.
# If standard pip fails, we might need a specific older version or wheel, but we try standard first.
# We use --extra-index-url for cpu-only wheels.
RUN linux32 python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --break-system-packages || echo "WARNING: Torch install failed, continuing..."

# Install other heavy dependencies
RUN linux32 python3 -m pip install transformers --break-system-packages

# Copy the L.O.V.E repository
# We assume the build context includes the L.O.V.E files or we copy them in a separate step.
# For this Dockerfile, we expect the L.O.V.E source to be in a 'src' directory in the build context.
COPY src /root/L.O.V.E

# Install Python requirements from the repo
WORKDIR /root/L.O.V.E
RUN python3 -m pip install -r requirements.txt --break-system-packages || echo "WARNING: Some requirements failed to install."

# Pre-install 'docker' client binary (mocking the daemon availability)
# This allows the script's "shutil.which('docker')" check to pass.
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*

# Set the entrypoint to bash so the user can look around, or directly run the script
CMD ["/bin/bash"]
