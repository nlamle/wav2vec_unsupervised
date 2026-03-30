# Use Debian 12 (Bookworm) to match the project's native environment
FROM debian:bookworm

# Prevent interactive prompts from freezing the setup
ENV DEBIAN_FRONTEND=noninteractive

# Install the absolute bare minimum required for your scripts to take over
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    wget \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to match your script's $INSTALL_ROOT expectation
WORKDIR /root/wav2vec_unsupervised

# Keep the container running interactively
CMD ["/bin/bash"]