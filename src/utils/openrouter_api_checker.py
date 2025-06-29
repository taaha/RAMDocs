from openai import OpenAI
from decouple import config

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
    model="meta-llama/llama-3.3-70b-instruct:free",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(completion.choices[0].message.content)
