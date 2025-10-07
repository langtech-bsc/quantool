# --- Stage 1: build llama.cpp (CUDA) ---
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS llama-builder

ENV DEBIAN_FRONTEND=noninteractive
# Ensure temp dirs are writable for _apt and tools
RUN mkdir -p /tmp /var/tmp && chmod 1777 /tmp /var/tmp
ENV TMPDIR=/var/tmp

# Build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates pkg-config \
    libcurl4-openssl-dev nvidia-cuda-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp.git
WORKDIR /opt/llama.cpp

# Build llama.cpp with CUDA
RUN cmake -B build -DGGML_CUDA=ON \
    && cmake --build build --config Release --parallel 4

# Collect binaries
# Collect artifacts: executables + shared libraries
RUN echo "=== Collecting llama.cpp artifacts ===" \
    && mkdir -p /opt/llama.cpp/dist/bin /opt/llama.cpp/dist/lib \
    # copy executables (llama-*)
    && find build -type f -name 'llama-*' -executable -exec cp {} /opt/llama.cpp/dist/bin/ \; \
    # copy shared libs from build/src and build/ggml/src (e.g., libllama.so, libggml*.so)
    && find build -type f -name 'libllama.so*' -exec cp {} /opt/llama.cpp/dist/lib/ \; || true \
    && find build -type f -name 'libggml*.so*' -exec cp {} /opt/llama.cpp/dist/lib/ \; || true \
    && ls -la /opt/llama.cpp/dist/bin/ /opt/llama.cpp/dist/lib/ \
    && strip /opt/llama.cpp/dist/bin/* 2>/dev/null || true

# --- Stage 2: runtime (Python 3.11 venv) ---
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Make sure temp space is writable for _apt
RUN mkdir -p /tmp /var/tmp && chmod 1777 /tmp /var/tmp
ENV TMPDIR=/var/tmp

# Base runtime deps + deadsnakes PPA (for Python 3.11 on 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git gnupg software-properties-common \
      libopenblas0 libgomp1 \
      build-essential pkg-config \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-distutils \
 && rm -rf /var/lib/apt/lists/*

# Create a dedicated 3.11 venv
RUN /usr/bin/python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:${PATH}"

# --- Pin the build toolchain to avoid 'Invalid version: cpython' ---
RUN python -m pip install --upgrade \
      "pip==24.0" \
      "packaging<25" \
      "setuptools==70.3.0" \
      "wheel==0.44.*" \
      "setuptools-scm>=8,<10" \
      build

# Sanity check
RUN python -V && pip -V && python -c "import packaging, setuptools, wheel; print(packaging.__version__, setuptools.__version__, wheel.__version__)"


# Copy llama.cpp tools built in stage 1
COPY --from=llama-builder /opt/llama.cpp/dist/bin/ /usr/local/bin/
COPY --from=llama-builder /opt/llama.cpp/convert_hf_to_gguf.py /usr/local/bin/convert_hf_to_gguf.py

# Copy libs
COPY --from=llama-builder /opt/llama.cpp/dist/bin/ /usr/local/bin/
COPY --from=llama-builder /opt/llama.cpp/dist/lib/ /usr/local/lib/

# Check
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}"
RUN printf "/usr/local/lib\n" > /etc/ld.so.conf.d/llama.conf && ldconfig || true

# App code
WORKDIR /app
COPY . .

# Install Python deps into the venv
RUN git clone https://github.com/langtech-bsc/quantool.git --single-branch --branch alpha-release \
    && cd quantool \
    && git fetch --tags --force \
    && pip install -U setuptools setuptools-scm wheel build \
    && pip install "git+https://github.com/NICTA/pyairports@master#egg=pyairports" \
    && pip install '.[all]'

# Optional: app-specific requirements
# RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    # pip cache purge || true

# Default CLI (override with `docker run ... quantool <args>`)
ENTRYPOINT ["quantool"]
CMD ["--help"]
