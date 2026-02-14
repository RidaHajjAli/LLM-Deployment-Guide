# llama.cpp Engine Setup

llama.cpp is a C++ implementation for running LLMs with minimal setup and efficient resource usage. This setup uses Docker to run llama.cpp with GPU acceleration.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- GGUF format model file

## Model Setup

1. Create a `models` directory in this folder if it doesn't exist
2. Download a GGUF model file and place it in the `models` directory
3. The default configuration expects `Llama-3.2-1B-Instruct-Q4_0.gguf`

To use a different model, update the `-m` parameter in the `docker-compose.yml` file.

You can find GGUF models on [Hugging Face](https://huggingface.co/models?library=gguf&apps=llama.cpp)

## Configuration

The `docker-compose.yml` file contains detailed configuration options:

- Port: `8082` (accessible at `http://localhost:8082`)
- Context length: 4096 tokens
- GPU layers: 99 (offloads all layers to GPU)
- Memory limit: 4GB (adjustable)
- Parallel sequences: 4 (configurable via N_PARALLEL environment variable)

You can customize these settings by modifying the command arguments in `docker-compose.yml`.

## Setup and Running

1. Ensure your model file is in the `models` directory

2. Start the llama.cpp service:
```bash
docker compose up --build
```

3. Wait for the service to be ready. The healthcheck will ensure the service is available (approximately 2 minutes).

4. The UI will be available at `http://localhost:8082`

## Testing

Run the test script to verify the setup:
```bash
python test_llama_cpp.py
```

The test script sends a request using the OpenAI-compatible API format.

## API Usage

The llama.cpp server provides an OpenAI-compatible API. Example using the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Llama-3.2-1B-Instruct-Q4_0.gguf",
    messages=[
        {"role": "user", "content": "Your question here"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Configuration Parameters

Key parameters in the docker-compose.yml:

- `-c 4096`: Context window size
- `-ngl 99`: Number of GPU layers
- `-np 4`: Number of parallel sequences
- `-fa on`: Enable Flash Attention
- `-b 2048`: Batch size for prompt processing
- `-ub 1024`: Batch size for generation
- `--cache-type-k q8_0`: Key cache quantization
- `--cache-type-v q8_0`: Value cache quantization
- `-t 8`: Number of CPU threads

## Customization

To change the model, update the `-m` parameter to point to your GGUF file in the models directory.

To adjust memory limits, modify the `memory` value under `deploy.resources.limits`.

To use a specific GPU, change `device_ids: ['0']` to your desired GPU ID.

## Stopping the Service

```bash
docker compose down
```

## Troubleshooting

If the container fails to start, check:
- The model file exists in the `models` directory
- The model path in docker-compose.yml matches your file name
- Your GPU has sufficient VRAM for the model
- NVIDIA Container Toolkit is properly installed
