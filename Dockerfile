# Use the official NVIDIA CUDA image (we continue with CUDA 12.1)
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04


# Install system packages, including pybind11-dev for the necessary headers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (you can still use conda for some dependencies if needed)
RUN curl -o /miniconda.sh -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /miniconda.sh && ./miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Update conda
RUN conda update -n base conda -y

# Install dependencies via conda that are not directly tied to torch
RUN conda install -y \
    python=3.9 \
    r-base \
    r-lme4 \
    r-lmerTest \
    r-emmeans \
    rpy2 \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    patsy \
    joblib \
    scipy \
    scikit-learn \
    jupyterlab \
    graphviz \
    python-graphviz \
 && conda clean -afy

# Instead of using conda for torch, install PyTorch and CUDA via pip
RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install PyG & its dependencies using wheels built for torch 2.3.1+cu121
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html \
    && pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cu121.html \
    && pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.3.1+cu121.html \
    && pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html \
    && pip install torch-geometric

# Install pymer4 from PyPI and additional packages
RUN pip install pymer4
RUN pip install -U ipywidgets
RUN pip install -U "ray[tune]"
RUN pip install -U "HEBO>=0.2.0"
RUN pip install openpyxl

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


