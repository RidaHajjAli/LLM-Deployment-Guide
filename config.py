from pydantic import BaseModel

class Settings(BaseModel):
    OLLAMA_BASE_URL: str = "http://localhost:11435"
    OLLAMA_MODEL: str = "llama3.2:1b"

    VLLM_URL: str = "http://localhost:8018"
    VLLM_MODEL: str = "casperhansen/llama-3.2-1b-instruct-awq"

    LLAMA_CPP_URL: str = "http://localhost:8082"
    LLAMA_CPP_MODEL: str = "Llama-3.2-1B-Instruct-Q4_0.gguf"

settings = Settings()

