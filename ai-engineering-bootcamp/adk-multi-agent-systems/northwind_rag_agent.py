"""
Northwind policy agent (Session 3 Path A) — Google ADK + real RAG tool.

Job: When a user asks a Northwind policy question, search ingested docs via the
Session 2 FastAPI retrieve endpoint, then answer with citations — or refuse.

Run:
  python northwind_rag_agent.py
  # or: streamlit run northwind_streamlit.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
APP_NAME = "northwind_rag_agent"


def normalize_rag_api_url(raw: str | None) -> str:
    """
    Accept either the API root or a pasted /docs URL.

    Correct: https://ai-internship-bnrf.onrender.com
    Also OK: .../docs  → stripped back to the API root
    """
    url = (raw or "https://ai-internship-bnrf.onrender.com").strip().rstrip("/")
    if url.endswith("/docs"):
        url = url[: -len("/docs")].rstrip("/")
    return url


RAG_API_URL = normalize_rag_api_url(
    os.getenv("RAG_API_URL") or os.getenv("API_BASE_URL")
)


def search_docs(query: str, top_k: int = 5) -> dict:
    """
    Search Northwind policy documents in the Session 2 vector store.

    Calls the live FastAPI GET /debug/retrieve endpoint (real HTTP tool — not a stub).
    Returns ranked chunks with document_id, chunk_index, score, and text.
    """
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "query must not be empty", "hits": []}

    k = max(1, min(int(top_k or 5), 10))
    url = f"{RAG_API_URL}/debug/retrieve?{urlencode({'q': q, 'k': k})}"
    try:
        # trust_env=False avoids local HTTP(S)_PROXY tunnel 403s (common in IDE sandboxes).
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - return observation, do not crash the agent
        return {
            "ok": False,
            "error": f"retrieve failed: {exc}",
            "rag_api_url": RAG_API_URL,
            "request_url": url,
            "hits": [],
        }

    hits = []
    for row in payload.get("hits") or []:
        text = (row.get("text") or "").strip()
        hits.append(
            {
                "document_id": row.get("document_id"),
                "chunk_index": row.get("chunk_index"),
                "chunk_id": row.get("point_id"),
                "score": row.get("score"),
                "text": text[:800],
            }
        )
    return {
        "ok": True,
        "query": payload.get("query", q),
        "top_k": payload.get("top_k", k),
        "hit_count": len(hits),
        "hits": hits,
    }


northwind_agent = Agent(
    name="northwind_policy_agent",
    model=MODEL,
    description="Answers Northwind employee-policy questions using retrieved handbook chunks.",
    instruction=(
        "You are the Northwind Robotics policy assistant.\n"
        "Goal: answer ONLY from retrieved Northwind documents.\n"
        "Always call search_docs before answering a policy question.\n"
        "Cite document_id (and chunk_index when available) for facts you use.\n"
        "If search_docs returns no useful hits, or the hits do not contain the answer, "
        'say exactly: I don\'t have enough information to answer that.\n'
        "Do not invent policies. Keep answers concise.\n"
        f"Stop after at most {MAX_AGENT_STEPS} tool calls; if still unsure, refuse.\n"
        "Done means: a cited answer from docs, or a clear refusal."
    ),
    tools=[search_docs],
)


def _part_debug(part: Any) -> list[dict]:
    """Map one content part to Think / Act / Observe log rows."""
    rows: list[dict] = []
    fc = getattr(part, "function_call", None)
    fr = getattr(part, "function_response", None)
    text = getattr(part, "text", None)

    if fc is not None:
        rows.append(
            {
                "phase": "Think",
                "detail": f"Decide to call tool `{fc.name}`",
            }
        )
        rows.append(
            {
                "phase": "Act",
                "detail": f"Call `{fc.name}` args={dict(fc.args or {})}",
            }
        )
    if fr is not None:
        response = fr.response
        if isinstance(response, dict):
            preview = json.dumps(response)[:400]
        else:
            preview = str(response)[:400]
        rows.append(
            {
                "phase": "Observe",
                "detail": f"Tool `{fr.name}` returned: {preview}",
            }
        )
    if text:
        rows.append({"phase": "Think", "detail": f"Draft/final text: {text[:400]}"})
    return rows


def label_events(events: list[Any]) -> list[dict]:
    """Convert ADK events into a simple Think → Act → Observe transcript."""
    steps: list[dict] = []
    for event in events:
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            steps.extend(_part_debug(part))
    return steps


async def run_agent(question: str) -> dict:
    """
    Run one multi-step agent task.

    Returns final_answer + steps[] (Think/Act/Observe) + metadata.
    Enforces a hard cap on tool-call events so the loop cannot run forever.
    """
    question = (question or "").strip()
    if not question:
        return {
            "final_answer": "",
            "steps": [],
            "error": "question must not be empty",
            "model": MODEL,
            "rag_api_url": RAG_API_URL,
        }

    service = InMemorySessionService()
    runner = Runner(agent=northwind_agent, app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=question)])

    events: list[Any] = []
    tool_calls = 0
    final_answer = "(no response)"

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=content,
    ):
        events.append(event)
        content_obj = getattr(event, "content", None)
        if content_obj and content_obj.parts:
            for part in content_obj.parts:
                if getattr(part, "function_call", None) is not None:
                    tool_calls += 1

        if tool_calls > MAX_AGENT_STEPS:
            final_answer = (
                "Stopped: max agent steps reached without a safe cited answer. "
                "I don't have enough information to answer that."
            )
            break

        if event.is_final_response() and content_obj and content_obj.parts:
            texts = [p.text for p in content_obj.parts if getattr(p, "text", None)]
            if texts:
                final_answer = "\n".join(texts)

    steps = label_events(events)
    return {
        "question": question,
        "final_answer": final_answer,
        "steps": steps,
        "tool_calls": tool_calls,
        "model": MODEL,
        "rag_api_url": RAG_API_URL,
        "max_agent_steps": MAX_AGENT_STEPS,
    }


async def main() -> None:
    tests = [
        "What is the remote work policy?",
        "What is the CEO's favorite pizza topping?",
    ]
    print(f"RAG_API_URL={RAG_API_URL}")
    print(f"MODEL={MODEL} MAX_AGENT_STEPS={MAX_AGENT_STEPS}")
    for q in tests:
        print(f"\n=== USER ===\n{q}\n")
        result = await run_agent(q)
        for step in result["steps"]:
            print(f"[{step['phase']}] {step['detail']}")
        print(f"\n=== FINAL ===\n{result['final_answer']}\n")


if __name__ == "__main__":
    asyncio.run(main())
