# Stage 1: Base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

ARG KOHYA_VERSION=v22.6.2

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Africa/Johannesburg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash \
    BATCH_SIZE=8 \
    S3_BUCKET_NAME=mytrainingdata \
    MIXED_PRECISION=fp16 \
    NUM_MACHINES=1 \
    MAX_TRAIN_EPOCHS=50 \
    DATASET_REPEATS=80 \
    OUTPUT_DIR="/workspace/output/"

# Install Ubuntu packages
RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
    software-properties-common \
    python3.10-venv \
    python3-pip \
    python3-tk \
    bash \
    dos2unix \
    git \
    ncdu \
    nginx \
    net-tools \
    openssh-server \
    libglib2.0-0 \
    libsm6 \
    libgl1 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    wget \
    curl \
    psmisc \
    rsync \
    vim \
    zip \
    unzip \
    p7zip-full \
    htop \
    screen \
    tmux \
    pkg-config \
    libcairo2-dev \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    apt-transport-https \
    ripgrep \
    fd-find \
    ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python


# Stage 2: Install kohya_ss and python modules
FROM base as kohya_ss_setup

# Add SDXL base model
# This needs to already have been downloaded:
#   wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
# RUN mkdir -p /sd-models
# COPY sd_xl_base_1.0.safetensors /sd-models/sd_xl_base_1.0.safetensors

# Create workspace working directory
WORKDIR /

# Install Kohya_ss
RUN git clone https://github.com/bmaltais/kohya_ss.git && \
    cd /kohya_ss && \
    git checkout ${KOHYA_VERSION}
WORKDIR /kohya_ss
COPY kohya_ss/requirements* ./
RUN python3 -m venv --system-site-packages venv && \
    source venv/bin/activate && \
    pip3 install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install xformers==0.0.22 \
    bitsandbytes==0.41.1 \
    tensorboard==2.14.1 \
    tensorflow==2.14.0 \
    wheel \
    scipy \
    tensorrt && \
    pip3 install -r requirements.txt && \
    pip3 install . && \
    deactivate

# Install Jupyter, gdown and OhMyRunPod
RUN pip3 install -U --no-cache-dir jupyterlab \
    jupyterlab_widgets \
    ipykernel \
    ipywidgets \
    gdown \
    OhMyRunPod

# Install RunPod File Uploader
RUN curl -sSL https://github.com/kodxana/RunPod-FilleUploader/raw/main/scripts/installer.sh -o installer.sh && \
    chmod +x installer.sh && \
    ./installer.sh

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# Install runpodctl
RUN wget https://github.com/runpod/runpodctl/releases/download/v1.13.0/runpodctl-linux-amd64 -O runpodctl && \
    chmod a+x runpodctl && \
    mv runpodctl /usr/local/bin

# Install Croc
RUN curl https://getcroc.schollz.com | bash

# Install awscli
RUN pip3 install awscli

# Set fd-find
RUN mv $(which fdfind) /usr/local/bin/fd

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

# NGINX Proxy
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/502.html /usr/share/nginx/html/502.html

WORKDIR /

# Set template version
ENV TEMPLATE_VERSION=1.12.5

# Copy the scripts
COPY --chmod=755 scripts/* ./

# Copy the accelerate configuration
COPY kohya_ss/accelerate.yaml ./

# Start the container
SHELL ["/bin/bash", "--login", "-c"]
CMD [ "/start.sh" ]
