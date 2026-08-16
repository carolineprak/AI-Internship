"""Week 1 live demo — five stages in one file, built up live in class."""

from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from openai import APIError, BadRequestError, NotFoundError, OpenAI
from pydantic import BaseModel, Field, ValidationError

from vector_store import delete_document_chunks, ingest_text, qdrant_healthcheck, retrieve

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}
ALLOWED_MODELS = frozenset(MODEL_PRICES_PER_1K)


def resolve_model(model: str | None) -> str:
    """Use default when model is omitted/blank; reject unknown ids with 400."""
    if model is None or not str(model).strip():
        return DEFAULT_MODEL
    chosen = str(model).strip()
    if chosen not in ALLOWED_MODELS:
        allowed = ", ".join(sorted(ALLOWED_MODELS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{chosen}'. Allowed: {allowed}",
        )
    return chosen


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class CitationSource(BaseModel):
    """One retrieved chunk available to ground the answer (stable /ask contract)."""

    document_id: str
    chunk_id: str
    chunk_index: int | None = None
    score: float | None = None


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    top_k: int = Field(default=5, ge=1, le=20)  # Session 2 RAG — retrieval depth


class AskResponse(BaseModel):
    """
    Stable /ask contract for Session 2+ (and Week 3 prep).

    Always present for both answers and refusals:
      question, answer, refused, sources, retrieved_chunk_ids,
      tokens_used, model, latency_ms, cost_usd
    """

    question: str
    answer: Answer
    refused: bool
    sources: list[CitationSource] = Field(default_factory=list)
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


REFUSAL_PHRASE = "I don't have enough information to answer that."

RAG_GROUNDING_PROMPT = """Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used.

Context:
{retrieved_chunks}

Question: {question}
"""


def format_retrieved_context(hits: list[dict]) -> str:
    """Render retrieved chunks for the grounding prompt."""
    if not hits:
        return "(No retrieved context.)"

    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        doc_id = hit.get("document_id") or "unknown"
        chunk_index = hit.get("chunk_index")
        point_id = hit.get("point_id") or ""
        score = hit.get("score")
        text = (hit.get("text") or "").strip()
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        blocks.append(
            f"[chunk {i} | document_id={doc_id} | chunk_index={chunk_index} "
            f"| point_id={point_id} | score={score_s}]\n{text}"
        )
    return "\n\n".join(blocks)


def build_grounding_prompt(question: str, hits: list[dict]) -> str:
    """Build the RAG user prompt: answer only from context, cite, or refuse."""
    return RAG_GROUNDING_PROMPT.format(
        retrieved_chunks=format_retrieved_context(hits),
        question=question.strip(),
    )


def retrieved_chunk_ids_from_hits(hits: list[dict]) -> list[str]:
    """Stable chunk IDs for the API response (prefer Qdrant point_id)."""
    ids: list[str] = []
    for hit in hits:
        point_id = hit.get("point_id")
        if point_id:
            ids.append(str(point_id))
            continue
        doc_id = hit.get("document_id") or "unknown"
        chunk_index = hit.get("chunk_index")
        ids.append(f"{doc_id}:{chunk_index}")
    return ids


def sources_from_hits(hits: list[dict]) -> list[CitationSource]:
    """Structured citations from retrieved chunks (same order as retrieval)."""
    sources: list[CitationSource] = []
    for hit in hits:
        doc_id = str(hit.get("document_id") or "unknown")
        point_id = hit.get("point_id")
        chunk_index = hit.get("chunk_index")
        chunk_id = str(point_id) if point_id else f"{doc_id}:{chunk_index}"
        score = hit.get("score")
        sources.append(
            CitationSource(
                document_id=doc_id,
                chunk_id=chunk_id,
                chunk_index=int(chunk_index) if chunk_index is not None else None,
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )
    return sources


def is_refusal_answer(answer_text: str) -> bool:
    """True when the model used the grounded refusal phrase."""
    return REFUSAL_PHRASE.lower() in (answer_text or "").lower()


class IngestRequest(BaseModel):
    """Body for POST /ingest — plain text plus a stable document id."""

    text: str
    document_id: str = Field(min_length=1)
    source: str | None = None  # optional filename / source label


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

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
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.get("/")
def root() -> RedirectResponse:
    """Send browsers to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    """Liveness check — process is up. No external deps, no secrets."""
    return {"status": "ok"}


@app.get("/debug/qdrant")
def debug_qdrant() -> dict:
    """Confirm Qdrant Cloud is reachable (no secrets returned)."""
    try:
        return qdrant_healthcheck()
    except Exception as exc:  # noqa: BLE001 - surface config/connectivity errors to the caller
        raise HTTPException(status_code=503, detail=f"Qdrant health check failed: {exc}") from exc


@app.get("/debug/retrieve")
def debug_retrieve(q: str, k: int = 5) -> dict:
    """Embed q and return top-k chunks with scores — no LLM call.

    Example:
      curl -s 'http://127.0.0.1:8000/debug/retrieve?q=remote%20work%20policy'
    """
    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="q must not be empty")
    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

    try:
        hits = retrieve(question, top_k=k, openai_client=client)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    return {
        "query": question,
        "top_k": k,
        "hits": hits,
    }


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and upsert a document into Qdrant Cloud.

    Example:
      curl -s -X POST http://127.0.0.1:8000/ingest \\
        -H "Content-Type: application/json" \\
        -d '{"text": "Remote work: up to 3 days per week with manager approval.", "document_id": "handbook", "source": "handbook.txt"}'
    """
    text = (body.text or "").strip()
    document_id = (body.document_id or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id must not be empty")

    try:
        result = ingest_text(
            text=text,
            document_id=document_id,
            source=body.source,
            openai_client=client,
        )
    except Exception as exc:  # noqa: BLE001 - return a clear API error
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc

    return IngestResponse(
        document_id=result["document_id"],
        chunks_indexed=result["chunks_indexed"],
        status=result["status"],
    )


@app.delete("/ingest/{document_id}")
def delete_ingest(document_id: str) -> dict:
    """Remove all chunks for a document_id (cleanup competing test docs)."""
    doc_id = (document_id or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="document_id must not be empty")
    try:
        delete_document_chunks(doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Delete failed: {exc}") from exc
    return {"document_id": doc_id, "status": "deleted"}


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """RAG ask: retrieve top-k chunks, ground the prompt, then Session 1 generation."""

    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    model = resolve_model(body.model)
    last_error: str | None = None
    start = time.perf_counter()

    try:
        hits = retrieve(question, top_k=body.top_k, openai_client=client)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    chunk_ids = retrieved_chunk_ids_from_hits(hits)
    sources = sources_from_hits(hits)
    grounded_prompt = build_grounding_prompt(question, hits)

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounded_prompt, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounded_prompt, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            return AskResponse(
                question=question,
                answer=answer,
                refused=is_refusal_answer(answer.answer),
                sources=sources,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=chunk_ids,
            )
        except HTTPException:
            raise
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue
        except (BadRequestError, NotFoundError) as exc:
            # OpenAI rejected the request (e.g. unknown model) — client error, not a 500.
            raise HTTPException(status_code=400, detail=f"Invalid model request: {exc}") from exc
        except APIError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream model error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - never leak an unhandled 500 to graders
            raise HTTPException(status_code=502, detail=f"Ask failed: {exc}") from exc

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
