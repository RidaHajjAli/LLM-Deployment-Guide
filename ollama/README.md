# Ollama Engine Setup

Ollama is a user-friendly platform for running LLMs locally. This setup uses Docker to run Ollama with GPU acceleration.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (optional, can run on CPU)
- NVIDIA Container Toolkit installed (if using GPU)

## Configuration

The `docker-compose.yml` file contains environment variables for optimization:

- Port: `11435` (mapped from container port 11434)
- Context length: 10000 tokens
- Parallel requests: 1
- Max loaded models: 1
- Flash Attention: Enabled
- GPU layers: 40

## Download the model

```bash
docker exec -it model-ollama sh
ollama pull llama3.2:1b
ollama list
exit
```

To increase the number of context of the model (by default it is 4096 tokens):
```bash
docker exec -it model-ollama sh
Modelfile="/tmp/Modelfile.txt"
echo "FROM llama3.2:1b" > $Modelfile
echo "PARAMETER num_ctx 12000" >> $Modelfile # Or how much you want, based on how much model supports
ollama create llama3.2_12k -f $Modelfile # Then change its name in config.py to be llama3.2_12k
ollama list # to confirm
exit # now the model is available


The default model is `llama3.2:1b` as configured in `config.py`.

## Setup and Running

1. Start the Ollama service:
```bash
docker compose up
```

2. Wait for the service to be ready (approximately 1 minute).

3. Pull the model (first time only):
```bash
docker exec -it model-ollama ollama pull llama3.2:1b
```

4. The API will be available at `http://localhost:11435`

## Testing

Run the test script to verify the setup:
```bash
python test_ollama.py
```

The test script sends a request using the OpenAI-compatible API format.

## API Usage

The Ollama server provides an OpenAI-compatible API. Example using the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[
        {"role": "user", "content": "Your question here"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Using Different Models

To use a different model:

1. Pull the model:
```bash
docker exec -it model-ollama ollama pull model-name
```

2. Update `config.py` to change the default model:
```python
OLLAMA_MODEL: str = "model-name"
```

Available models can be found at: https://ollama.com/library

## Configuration Options

Environment variables in `docker-compose.yml`:

- `OLLAMA_NUM_CTX`: Context window size
- `OLLAMA_NUM_PARALLEL`: Number of parallel requests
- `OLLAMA_MAX_LOADED_MODELS`: Maximum models to keep in memory
- `OLLAMA_FLASH_ATTENTION`: Enable Flash Attention (1 for on, 0 for off)
- `OLLAMA_NUM_GPU_LAYERS`: Number of layers to offload to GPU

## Running Without GPU

If you don't have a GPU, remove or comment out the `deploy.resources.reservations.devices` section in `docker-compose.yml`.

## Customization

To adjust memory limits, uncomment and set the `mem_limit` and `memswap_limit` values in `docker-compose.yml`.

To use multiple GPUs, change `count: all` to a specific number or keep `all` to use all available GPUs.

## Stopping the Service

```bash
docker compose down
```

To remove downloaded models as well:
```bash
docker compose down -v
```

## Managing Models

List installed models:
```bash
docker exec -it model-ollama ollama list
```

Remove a model:
```bash
docker exec -it model-ollama ollama rm model-name
```

## Troubleshooting

If the container fails to start:
- Check if port 11435 is available
- Verify NVIDIA Container Toolkit is installed (for GPU usage)
- Check Docker logs: `docker logs model-ollama`
