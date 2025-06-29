from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from decouple import config

API_KEY = config("OPENROUTER_API_KEY")


def call_llm_pydantic_ai(
    prompt: str,
    model_id: str = "meta-llama/llama-3.3-70b-instruct:free",
    max_new_tokens: int = 512,
    pydantic_model: BaseModel = None,
) -> str:
    model = OpenAIModel(
        model_id,
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1", api_key=API_KEY
        ),
    )
    agent = Agent(model=model, output_type=pydantic_model, retries=10, output_retries=10)
    return agent.run_sync(prompt)
