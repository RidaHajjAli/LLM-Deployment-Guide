# Builder Stage
# Change the cuda and ubuntu based on your versions
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder 

# You can change the tag to choose another one from here: https://github.com/ggml-org/llama.cpp/releases
ARG LLAMA_TAG=b7531 
ENV DEBIAN_FRONTEND=noninteractive \
    # Based on your Cuda
    CUDA_ARCH=89 

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone specific tag
RUN git clone --depth 1 --branch ${LLAMA_TAG} https://github.com/ggml-org/llama.cpp.git . && \
    echo "Building tag: ${LLAMA_TAG}"

# Build llama-server with static linking to avoid shared library dependencies
RUN mkdir build && cd build && \
    cmake .. \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF && \
    cmake --build . --config Release -j$(nproc) --target llama-server

# Runtime Stage
# Change the cuda and ubuntu based on your versions
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install minimal runtime dependencies (including libgomp for OpenMP support)
RUN apt-get update && apt-get install -y \
    curl \
    libcurl4 \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the statically linked binary (no .so files needed)
COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server

EXPOSE 8080

ENTRYPOINT ["llama-server"]