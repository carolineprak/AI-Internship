"""Qdrant Cloud vector store helpers for Session 2 RAG.

Config is loaded from environment variables only — never hardcode secrets.
Embedding model is locked to text-embedding-3-small (1536 dims) so ingest
and query always use the same vector space.
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
_POINT_NAMESPACE = uuid.UUID("8f3c2e1a-6b4d-4a9f-9c1e-2d7a5b8e0f11")


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set it in week-1/.env locally and in Render → Environment."
        )
    return value


def get_collection_name() -> str:
    return (os.getenv("QDRANT_COLLECTION") or "rag_demo").strip()


def get_chunk_settings() -> tuple[int, int]:
    """Chunk size/overlap from env (defaults tuned for policy-style docs)."""
    chunk_size = int(os.getenv("CHUNK_SIZE") or "600")
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP") or "100")
    if chunk_size < 1:
        raise ValueError("CHUNK_SIZE must be >= 1")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE")
    return chunk_size, chunk_overlap


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Shared Qdrant Cloud client (URL + API key from env)."""
    return QdrantClient(
        url=_require_env("QDRANT_URL"),
        api_key=_require_env("QDRANT_API_KEY"),
        check_compatibility=False,
    )


def get_openai_client() -> OpenAI:
    """OpenAI client for embeddings (uses OPENAI_API_KEY)."""
    return OpenAI()


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """Embed texts with the locked Session 2 model."""
    if not texts:
        return []
    openai_client = client or get_openai_client()
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # API returns data sorted by index, but sort defensively.
    ordered = sorted(response.data, key=lambda row: row.index)
    return [row.embedding for row in ordered]


def embed_query(text: str, client: OpenAI | None = None) -> list[float]:
    return embed_texts([text], client=client)[0]


def ensure_collection(client: QdrantClient | None = None, collection: str | None = None) -> str:
    """Create the collection if missing (cosine, 1536 dims for text-embedding-3-small)."""
    qdrant = client or get_qdrant_client()
    name = collection or get_collection_name()
    existing = {c.name for c in qdrant.get_collections().collections}
    if name not in existing:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIMS, distance=qmodels.Distance.COSINE
            ),
        )

    # Filtered deletes/searches by document_id need a keyword payload index on Qdrant Cloud.
    try:
        qdrant.create_payload_index(
            collection_name=name,
            field_name="document_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        # Index may already exist — safe to continue.
        pass
    return name


_SECTION_BANNER = "=============================================================================="


def _split_handbook_sections(text: str) -> list[tuple[str, str]]:
    """
    Split POL-style docs on ==== banners into (section_title, body) pairs.

    Title is the non-empty line immediately before a banner (e.g. "4. REMOTE WORK...").
    Content before the first banner is kept as ("", preamble).
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    preamble: list[str] = []
    title = ""
    body: list[str] = []
    i = 0
    saw_banner = False

    while i < len(lines):
        line = lines[i]
        if line.strip() == _SECTION_BANNER:
            saw_banner = True
            # Flush anything accumulated before this banner as preamble/previous body.
            if body or (not sections and preamble):
                blob = "\n".join(preamble + body).strip() if not sections else "\n".join(body).strip()
                if not sections and preamble and not body:
                    blob = "\n".join(preamble).strip()
                if blob:
                    sections.append((title, blob))
                preamble = []
                body = []

            # Title is the last non-empty line before this banner (already in body or prev).
            # Pattern in Northwind: TITLE\n====\nBODY...\n====  OR  ====\nTITLE\n====\nBODY
            # Prefer title between two banners: skip opening banner, next non-empty = title,
            # then optional closing banner, then body until next banner.
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].strip() != _SECTION_BANNER:
                title = lines[i].strip()
                i += 1
            # Optional second banner under the title
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].strip() == _SECTION_BANNER:
                i += 1
            body = []
            while i < len(lines) and lines[i].strip() != _SECTION_BANNER:
                body.append(lines[i])
                i += 1
            blob = "\n".join(body).strip()
            if blob or title:
                sections.append((title, blob))
            title = ""
            body = []
            continue

        if not saw_banner:
            preamble.append(line)
        else:
            body.append(line)
        i += 1

    if not saw_banner:
        blob = text.strip()
        return [("", blob)] if blob else []

    trailing = "\n".join(body).strip()
    if trailing:
        sections.append((title, trailing))
    return [(t, b) for t, b in sections if b.strip()]


def chunk_text(text: str) -> list[str]:
    """
    Section-aware chunking for policy handbooks.

    Prefer splitting on ==== section banners, then size-split within a section
    while prepending the section title to each chunk so embeddings keep topic context.
    Falls back to recursive character splitting for plain text.
    """
    chunk_size, chunk_overlap = get_chunk_settings()
    within_section = RecursiveCharacterTextSplitter(
        chunk_size=max(200, chunk_size - 80),  # leave room for title prefix
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if _SECTION_BANNER in text:
        chunks: list[str] = []
        for title, body in _split_handbook_sections(text):
            prefix = f"{title}\n\n" if title else ""
            # If whole section fits, keep it as one chunk (best for leave / remote answers).
            candidate = f"{prefix}{body}".strip()
            if len(candidate) <= chunk_size:
                chunks.append(candidate)
                continue
            for piece in within_section.split_text(body):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(f"{prefix}{piece}".strip() if prefix else piece)
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def _point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


def delete_document_chunks(
    document_id: str,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> None:
    """Remove prior chunks for a document_id so re-ingest does not leave orphans."""
    qdrant = client or get_qdrant_client()
    name = collection or get_collection_name()
    qdrant.delete(
        collection_name=name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )


def ingest_text(
    text: str,
    document_id: str,
    source: str | None = None,
    openai_client: OpenAI | None = None,
) -> dict:
    """
    Chunk → embed → upsert one document into Qdrant.

    Payload metadata per point: document_id, chunk_index, source, text.
    """
    qdrant = get_qdrant_client()
    collection = ensure_collection(qdrant)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No chunks produced from text")

    vectors = embed_texts(chunks, client=openai_client)
    source_value = (source or "").strip() or document_id

    # Drop previous chunks for this document_id, then write the new set.
    delete_document_chunks(document_id, client=qdrant, collection=collection)

    points = [
        qmodels.PointStruct(
            id=_point_id(document_id, index),
            vector=vector,
            payload={
                "document_id": document_id,
                "chunk_index": index,
                "source": source_value,
                "text": chunk,
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    qdrant.upsert(collection_name=collection, points=points)

    chunk_size, chunk_overlap = get_chunk_settings()
    return {
        "document_id": document_id,
        "chunks_indexed": len(points),
        "status": "ok",
        "collection": collection,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL,
    }


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    openai_client: OpenAI | None = None,
) -> list[dict]:
    """
    Embed the question and return top-k chunks with scores (no LLM).

    Each hit includes: text, document_id, chunk_index, source, score, point_id.
    """
    qdrant = get_qdrant_client()
    collection = ensure_collection(qdrant)
    vector = embed_query(query, client=openai_client)

    results = qdrant.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )

    hits: list[dict] = []
    for point in results.points:
        payload = point.payload or {}
        hits.append(
            {
                "point_id": str(point.id),
                "score": float(point.score) if point.score is not None else None,
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "source": payload.get("source"),
                "text": payload.get("text"),
            }
        )
    return hits


def qdrant_healthcheck() -> dict:
    """
    Confirm Qdrant Cloud is reachable and report collection status.

    Safe to call from a debug route — does not print secrets.
    """
    qdrant = get_qdrant_client()
    collection = get_collection_name()
    collections = qdrant.get_collections().collections
    names = sorted(c.name for c in collections)
    exists = collection in names

    points_count: int | None = None
    if exists:
        info = qdrant.get_collection(collection)
        points_count = int(info.points_count or 0)

    chunk_size, chunk_overlap = get_chunk_settings()
    return {
        "ok": True,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dims": EMBEDDING_DIMS,
        "collection": collection,
        "collection_exists": exists,
        "points_count": points_count,
        "collections": names,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
