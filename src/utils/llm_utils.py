from decouple import config
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)


def call_llm(
    prompt: str,
    model_id: str = "meta-llama/llama-3.3-70b-instruct:free",
    max_new_tokens: int = 128,
) -> str:
    completion = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    return completion.choices[0].message.content
