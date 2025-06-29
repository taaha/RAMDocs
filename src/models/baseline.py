from pydantic import BaseModel, Field


class Answer(BaseModel):
    all_correct_answers: list[str] = Field(description="The correct answers to the question.")
    explanation: str = Field(description="The explanation of the correct answers. Please provide a step-by-step reasoning explanation.")
