"""Stage 4 server — selectable model and latency readout.

Run: uvicorn serve_stage4:app --port 8000 --reload
"""

from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()
client = OpenAI()

DEFAULT_MODEL = "gpt-4o"


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str
    force_bad: bool = False
    model: str | None = None


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int


def call_structured(question: str, model: str) -> tuple[Answer, int]:
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    return parsed, tokens_used


def call_unsafe(question: str, model: str) -> tuple[Answer, int]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )
    raw = completion.choices[0].message.content or ""
    answer = Answer.model_validate_json(raw)
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    return answer, tokens_used


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    model = body.model or DEFAULT_MODEL
    last_error: str | None = None

    for attempt in range(2):
        try:
            start = time.perf_counter()
            if body.force_bad and attempt == 0:
                answer, tokens_used = call_unsafe(body.question, model)
            else:
                answer, tokens_used = call_structured(body.question, model)
            latency_ms = int((time.perf_counter() - start) * 1000)
            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
