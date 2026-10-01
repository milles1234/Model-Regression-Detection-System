"""
Eval test runner
================

``run_evaluation`` is the public entry-point.  It:

1. Accepts a ``PromptConfig`` and the golden dataset (list of test-case dicts
   loaded from ``tests/data.json``).
2. Runs every test case through the LLM classifier asynchronously.
3. Throttles concurrency via an ``asyncio.Semaphore`` to avoid API rate-limit
   errors and keep costs predictable.
4. Scores each test case on **four independent dimensions**:

   - ``category_score``  – binary 0/1 exact-match on the predicted category.
   - ``summary_score``   – LLM-as-judge rating 1–5 measuring how well the
                           generated summary captures the ground-truth summary.
   - ``elapsed_seconds`` – wall-clock latency for the classify + judge calls.
   - ``input_tokens`` / ``output_tokens`` / ``total_tokens`` – token usage
                           reported by the classifier API call.

5. Returns a list of ``EvalResult`` objects -- one per test case -- with all
   dimensions populated so a downstream reporter can aggregate them.

Usage example
-------------
::

    import asyncio, json
    from src.prompt_loader import load_prompt
    from evals.runner import run_evaluation

    dataset = json.loads(open("tests/data.json").read())["test_cases"]
    config  = load_prompt("prompts/v1.yaml")
    results = asyncio.run(run_evaluation(config, dataset))
"""

# pyrefly: ignore [invalid-syntax]
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types

from src.classifier import classify_email, DEFAULT_MODEL
from src.models import ClassificationResponse, PromptConfig

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Maximum number of in-flight LLM requests at any moment.
DEFAULT_CONCURRENCY = 10

#: Exponential back-off base in seconds (attempt ** base).
RETRY_BACKOFF_BASE = 1.5

#: Maximum number of automatic retries per test case before recording an error.
MAX_RETRIES = 3

#: Model used for the LLM-as-judge summary scoring call.
JUDGE_MODEL = "gemini-3.8-flash"

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Holds one test case's input, expected answer, and all scoring dimensions."""

    #: Test-case identifier (e.g. "TC-001")
    id: str

    #: Original email text sent to the model
    input: str

    #: Ground-truth answer from the golden dataset
    expected_category: str
    expected_summary: str

    #: Difficulty tag from the dataset ("easy" / "medium" / "hard")
    difficulty: str

    #: Raw output from the LLM (None if the call failed after all retries)
    actual: Optional[dict] = None

    #: Error message if the call ultimately failed
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Scoring dimensions (all populated by _run_single)
    # ------------------------------------------------------------------

    #: Dimension 1 – Exact category match: 1 if correct, 0 otherwise.
    category_score: int = field(default=0)

    #: Dimension 2 – LLM-as-judge summary relevance rating: 1 (poor) – 5 (excellent).
    #:               Set to 0 when the classify call itself failed.
    summary_score: int = field(default=0)

    #: Dimension 3 – Wall-clock seconds from first API call to all scoring done.
    elapsed_seconds: float = field(default=0.0)

    #: Dimension 4 – Token usage reported by the classifier API call.
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    total_tokens: int = field(default=0)


# ---------------------------------------------------------------------------
# LLM-as-judge: summary scoring
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for an email classification system.
Your task is to rate how well a generated summary captures the key information
in a ground-truth (reference) summary on a scale from 1 to 5.

Rubric:
  5 – The generated summary is semantically equivalent to the reference.
      All key facts (issue type, action requested, identifiers) are present.
  4 – Minor wording differences; all key facts are present.
  3 – Most key facts are present but at least one is missing or vague.
  2 – Only partial information; the core issue is mentioned but important
      details are missing.
  1 – The generated summary is largely irrelevant or incorrect.

Respond with a single integer between 1 and 5. No explanation, just the digit.
"""

_JUDGE_USER_TEMPLATE = """\
Reference summary:
{reference}

Generated summary:
{generated}
"""


async def _score_summary(
    reference: str,
    generated: str,
    client: genai.Client,
    semaphore: asyncio.Semaphore,
) -> int:
    """Use an LLM-as-judge to rate the generated summary against the reference.

    Parameters
    ----------
    reference:
        Ground-truth summary from the golden dataset.
    generated:
        Summary produced by the classifier for this test case.
    client:
        Authenticated ``google.genai.Client`` instance (shared with classify).
    semaphore:
        Shared concurrency gate so judge calls count against the same limit.

    Returns
    -------
    int
        A rating in [1, 5].  Falls back to 0 on any parse or API error.
    """
    prompt_text = _JUDGE_USER_TEMPLATE.format(
        reference=reference.strip(),
        generated=generated.strip(),
    )

    async with semaphore:
        try:
            response = await client.aio.models.generate_content(
                model=JUDGE_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=prompt_text)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_JUDGE_SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            raw = (response.text or "").strip()
            # Accept the first digit character found
            for ch in raw:
                if ch.isdigit():
                    score = int(ch)
                    return max(1, min(5, score))  # clamp to [1, 5]
        except Exception:  # noqa: BLE001
            pass

    return 0  # fallback on error


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run_single(
    test_case: dict,
    prompt_config: PromptConfig,
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    model: str,
) -> EvalResult:
    """Run one test case with retry + exponential back-off, then score it.

    Scoring dimensions computed here
    ---------------------------------
    category_score  : 1 if predicted category matches expected, else 0.
    summary_score   : LLM-as-judge rating 1-5 via ``_score_summary``.
    elapsed_seconds : Wall-clock time covering classify + judge calls.
    input_tokens    : Tokens consumed by the classifier prompt.
    output_tokens   : Tokens in the classifier's response.
    total_tokens    : input_tokens + output_tokens (as reported by the API).
    """
    tc_id: str = test_case["id"]
    email_text: str = test_case["input"]
    expected = test_case["expected_output"]
    difficulty: str = test_case.get("expected_difficulty", "unknown")

    result = EvalResult(
        id=tc_id,
        input=email_text,
        expected_category=expected["category"],
        expected_summary=expected["summary"],
        difficulty=difficulty,
    )

    start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Step 1: Classify the email (with retry + back-off)                  #
    # ------------------------------------------------------------------ #
    last_error: str = ""
    classify_response: Optional[ClassificationResponse] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                classify_response = await classify_email(
                    email_text=email_text,
                    prompt_config=prompt_config,
                    client=client,
                    model=model,
                )
            break  # success — exit retry loop

        except Exception as exc:  # noqa: BLE001
            last_error = (
                f"[attempt {attempt}/{MAX_RETRIES}] "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
    else:
        # All retries exhausted — record the error and skip scoring
        result.error = last_error
        result.elapsed_seconds = time.perf_counter() - start
        return result

    # ------------------------------------------------------------------ #
    # Step 2: Store the raw response and token usage (dimension 4)        #
    # ------------------------------------------------------------------ #
    result.actual = classify_response.model_dump() if classify_response else None

    usage = classify_response.usage
    result.input_tokens = usage.input_tokens
    result.output_tokens = usage.output_tokens
    result.total_tokens = usage.total_tokens

    # ------------------------------------------------------------------ #
    # Step 3: Dimension 1 — Exact category match                         #
    # ------------------------------------------------------------------ #
    result.category_score = int(
        classify_response.result.category.strip().lower()
        == result.expected_category.strip().lower()
    )

    # ------------------------------------------------------------------ #
    # Step 4: Dimension 2 — LLM-as-judge summary score (1-5)             #
    # ------------------------------------------------------------------ #
    result.summary_score = await _score_summary(
        reference=result.expected_summary,
        generated=classify_response.result.summary,
        client=client,
        semaphore=semaphore,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Dimension 3 — Total elapsed wall-clock time                #
    # ------------------------------------------------------------------ #
    result.elapsed_seconds = time.perf_counter() - start

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_evaluation(
    prompt_config: PromptConfig,
    golden_dataset: list[dict],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[EvalResult]:
    """Run every test case in *golden_dataset* through the LLM classifier
    and score each result on four independent dimensions.

    Parameters
    ----------
    prompt_config:
        Versioned prompt produced by ``src.prompt_loader.load_prompt``.
    golden_dataset:
        List of test-case dicts (keys: ``id``, ``input``,
        ``expected_output``, ``expected_difficulty``).
    api_key:
        Google AI API key.  Falls back to the ``GEMINI_API_KEY``
        environment variable when omitted.
    model:
        Gen AI model ID (default: ``gemini-2.0-flash``).
    concurrency:
        Maximum simultaneous in-flight API calls (default: 10).
        Applies to both classify *and* judge calls so total in-flight
        requests never exceed this value.

    Returns
    -------
    list[EvalResult]
        One ``EvalResult`` per test case in golden-dataset order.
        Every result carries:

        * ``category_score``  – 0 or 1 (exact match)
        * ``summary_score``   – 1-5 (LLM-as-judge; 0 on error)
        * ``elapsed_seconds`` – total wall-clock latency
        * ``input_tokens``, ``output_tokens``, ``total_tokens``
    """
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "No API key provided. Pass api_key= or set GEMINI_API_KEY."
        )

    client = genai.Client(api_key=resolved_key)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _run_single(
            test_case=tc,
            prompt_config=prompt_config,
            client=client,
            semaphore=semaphore,
            model=model,
        )
        for tc in golden_dataset
    ]

    # gather preserves input order; _run_single never raises so no need for
    # return_exceptions=True.
    results: list[EvalResult] = await asyncio.gather(*tasks)
    return list(results)
