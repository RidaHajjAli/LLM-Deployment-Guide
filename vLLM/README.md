# vLLM Engine Setup

vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. This setup uses Docker to run vLLM with GPU acceleration.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- HuggingFace account and access token

## Configuration

1. Create a `.env` file in this directory with your HuggingFace token:
```
HF_TOKEN=your_huggingface_token_here
```

2. The default configuration uses:
   - Model: `casperhansen/llama-3.2-1b-instruct-awq`
   - Port: `8018` (mapped from container port 8000)
   - GPU memory utilization: 70%
   - Max model length: 4096 tokens

## Setup and Running

1. Start the vLLM service:
```bash
docker compose up --build
```

2. Wait for the service to be ready. The healthcheck will ensure the service is available before marking it as healthy (approximately 2 minutes).

3. The API will be available at `http://localhost:8018/v1`

## Testing

Run the test script to verify the setup:
```bash
python test_vLLM.py
```

The test script sends a request using the OpenAI-compatible API format.

## API Usage

The vLLM server provides an OpenAI-compatible API. Example using the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8018/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="casperhansen/llama-3.2-1b-instruct-awq",
    messages=[
        {"role": "user", "content": "Your question here"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Customization

To use a different model, modify the `--model` parameter in the `docker-compose.yml` file under the `command` section.

To adjust GPU memory usage, modify the `--gpu-memory-utilization` parameter (value between 0 and 1).

## Stopping the Service

```bash
docker compose down
```

To remove the downloaded models as well:
```bash
docker compose down -v
```
