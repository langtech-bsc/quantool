# --- Stage 1: build llama.cpp (CUDA + cuBLAS) ---
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS llama-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp.git

# Build CLI + quantizer with CUDA (examples must be ON for these tools)
WORKDIR /opt/llama.cpp/build
RUN cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DLLAMA_BUILD_EXAMPLES=ON \
      -DLLAMA_BUILD_SERVER=OFF \
 && cmake --build . --config Release --parallel \
           --target llama-cli llama-quantize \
 && strip bin/* || true

# --- Stage 2: runtime + library ---
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libopenblas0 libgomp1 ca-certificates git curl python3 python3-pip python3-venv \
  && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Copy just the built tools
COPY --from=llama-builder /opt/llama.cpp/build/bin/llama-* /usr/local/bin/
COPY --from=llama-builder /opt/llama.cpp/convert_hf_to_gguf.py /usr/local/bin/convert_hf_to_gguf.py

WORKDIR /app
COPY . .

RUN git clone https://github.com/langtech-bsc/quantool.git --branch higgs && \
    cd quantool && \
    pip install -e '.[all]'

RUN python -m pip install -U pip setuptools wheel && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip cache purge || true

ENTRYPOINT ["quantool"]
CMD ["--help"]