"""
Email classification feature — wraps the Google Gen AI SDK.

The public surface is a single coroutine:

    response = await classify_email(email_text, prompt_config, client)

It returns a ``ClassificationResponse`` containing a validated Pydantic
``ClassificationResult`` and a ``ClassificationUsage`` with token counts,
so callers never have to touch raw LLM output.
"""

# pyrefly: ignore [invalid-syntax]
from __future__ import annotations

import json
import re

from google import genai
from google.genai import types

from src.models import ClassificationResult, ClassificationResponse, ClassificationUsage, PromptConfig

# ---------------------------------------------------------------------------
# Module-level default model.  Callers can override via keyword argument.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.8-flash"


def _build_messages(
    email_text: str,
    prompt_config: PromptConfig,
) -> list[types.Content]:
    """Convert a PromptConfig + raw email into a Gen AI ``Content`` list."""
    contents: list[types.Content] = []

    # Few-shot examples (stored as {role, content} dicts in the YAML)
    for ex in prompt_config.examples:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=ex.get("input", ""))],
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part(text=json.dumps(ex.get("output", {})))],
            )
        )

    # Actual email under evaluation
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=email_text)],
        )
    )

    return contents


def _parse_result(raw: str) -> ClassificationResult:
    """Extract and validate a ``ClassificationResult`` from raw LLM text.

    The model is instructed to respond with JSON, but may wrap it in a
    markdown code-fence.  We strip the fence and parse.
    """
    # Strip optional ```json ... ``` fence
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    data = json.loads(cleaned.strip())
    return ClassificationResult(**data)


async def classify_email(
    email_text: str,
    prompt_config: PromptConfig,
    client: genai.Client,
    model: str = DEFAULT_MODEL,
) -> ClassificationResponse:
    """Classify a single customer email using the given ``PromptConfig``.

    Parameters
    ----------
    email_text:
        Raw customer email body.
    prompt_config:
        Versioned prompt config (system prompt + optional few-shot examples).
    client:
        An authenticated ``google.genai.Client`` instance.
    model:
        Gen AI model ID to use (defaults to ``gemini-2.0-flash``).

    Returns
    -------
    ClassificationResponse
        Contains a Pydantic-validated ``ClassificationResult``
        (``category`` + ``summary``) and a ``ClassificationUsage``
        with ``input_tokens``, ``output_tokens``, and ``total_tokens``.
    """
    system_instruction = (
        prompt_config.system_prompt
        + "\n\nRespond ONLY with a JSON object with exactly two keys: "
        '"category" and "summary". Do not include any other text.'
    )

    contents = _build_messages(email_text, prompt_config)

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,  # deterministic for eval reproducibility
            response_mime_type="application/json",
        ),
    )

    raw_text: str = response.text or ""
    classification = _parse_result(raw_text)

    usage_meta = response.usage_metadata
    usage = ClassificationUsage(
        input_tokens=usage_meta.prompt_token_count or 0,
        output_tokens=usage_meta.candidates_token_count or 0,
        total_tokens=usage_meta.total_token_count or 0,
    )

    return ClassificationResponse(result=classification, usage=usage)
