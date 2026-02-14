from openai import OpenAI
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import settings

client = OpenAI(
    base_url=f"{settings.LLAMA_CPP_URL}/v1",
    api_key="EMPTY"
)

def test_completion():
    response = client.chat.completions.create(
        model=settings.LLAMA_CPP_MODEL,
        messages=[
            {"role": "user", "content": "Explain transformers in one paragraph"}
        ],
        max_tokens=1000,
        temperature=0.7,
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(f"{chunk.choices[0].delta.content}", end="", flush=True)

if __name__ == "__main__":
    try:
        test_completion()
    except Exception as e:
        print(f"Error: {e}")
