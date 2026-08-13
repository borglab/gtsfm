# syntax=docker/dockerfile:1.7

# This image contains only the stable system and Python dependency layers.
# GTSFM source and third-party model code are mounted by Modal at deploy time,
# so normal application changes do not rebuild this multi-gigabyte layer.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.8.13

ENV DEBIAN_FRONTEND=noninteractive \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    PATH=/root/.local/bin:${PATH}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        graphviz \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libx11-6 \
        libzstd-dev \
        ninja-build \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && uv python install ${PYTHON_VERSION} \
    && uv venv --python ${PYTHON_VERSION} /opt/gtsfm-venv

ENV VIRTUAL_ENV=/opt/gtsfm-venv \
    UV_PROJECT_ENVIRONMENT=/opt/gtsfm-venv \
    PATH=/opt/gtsfm-venv/bin:/root/.local/bin:${PATH}

WORKDIR /opt/gtsfm-runtime
COPY pyproject.toml uv.lock ./

RUN uv sync \
        --frozen \
        --no-dev \
        --no-install-project \
    && python -c "import fastapi, gsplat, gtsam, spz, torch; print(torch.__version__)"

# Volume-backed cache locations are assigned by visualization/modal_app.py at
# container startup. They must not be present while Modal extends this image,
# because build tools can otherwise populate the future Volume mount target.
ENV PYTHONPATH=/root

WORKDIR /root
