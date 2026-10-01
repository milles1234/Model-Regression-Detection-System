from typing import Literal, List
from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    category: Literal["billing", "technical", "account", "general"]
    summary: str = Field(description="A one-sentence summary of the email.")


class PromptConfig(BaseModel):
    version: str
    system_prompt: str
    examples: List[dict] = []


class ClassificationUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ClassificationResponse(BaseModel):
    result: ClassificationResult
    usage: ClassificationUsage