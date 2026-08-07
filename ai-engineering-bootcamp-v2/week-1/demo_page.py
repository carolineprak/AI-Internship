"""Single test page for all five /ask demo stages.

Run this page:
  streamlit run demo_page.py
"""

from __future__ import annotations

import json

import httpx
import streamlit as st

WORKDIR_CMD = "ai-engineering-bootcamp-v2/week-1"  # path from repo root

STAGES = [
    {
        "num": 1,
        "title": "Bare /ask",
        "serve": "uvicorn serve_stage1:app --port 8000 --reload",
        "look_for": "Plain `answer` string and real `tokens_used`.",
        "dummy_question": "What is Retrieval-Augmented Generation in one sentence?",
        "fields": [],
    },
    {
        "num": 2,
        "title": "Structured output",
        "serve": "uvicorn serve_stage2:app --port 8000 --reload",
        "look_for": "`answer` is an object with `confidence` and `sources_needed`.",
        "dummy_question": "Explain what an embedding is in one sentence.",
        "fields": [],
    },
    {
        "num": 3,
        "title": "Guardrail + retry",
        "serve": "uvicorn serve_stage3:app --port 8000 --reload",
        "look_for": "Normal question works; `force_bad` triggers retry then succeeds.",
        "dummy_question": "What is a vector database?",
        "dummy_force_bad": True,
        "fields": ["force_bad"],
    },
    {
        "num": 4,
        "title": "Model selectable",
        "serve": "uvicorn serve_stage4:app --port 8000 --reload",
        "look_for": "`model` and `latency_ms` in the response; swap models live.",
        "dummy_question": "What is chunking in RAG?",
        "dummy_model": "gpt-4o-mini",
        "fields": ["force_bad", "model"],
    },
    {
        "num": 5,
        "title": "Cost readout",
        "serve": "uvicorn serve_stage5:app --port 8000 --reload",
        "look_for": "`cost_usd` closes the loop — same prompt, different model, different cost.",
        "dummy_question": "What is Retrieval-Augmented Generation in one sentence?",
        "dummy_model": "gpt-4o",
        "fields": ["force_bad", "model"],
    },
]


def build_payload(
    question: str,
    stage: dict,
    force_bad: bool,
    model: str | None,
) -> dict:
    payload: dict = {"question": question}
    if "force_bad" in stage["fields"]:
        payload["force_bad"] = force_bad
    if "model" in stage["fields"] and model:
        payload["model"] = model
    return payload


def call_ask(base_url: str, payload: dict) -> tuple[int, dict | str]:
    try:
        response = httpx.post(f"{base_url.rstrip('/')}/ask", json=payload, timeout=120.0)
        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {base_url} — start the stage server first."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def render_curl(base_url: str, payload: dict) -> str:
    body = json.dumps(payload)
    return (
        f'curl -s -X POST {base_url.rstrip("/")}/ask '
        f'-H "Content-Type: application/json" -d \'{body}\''
    )


def render_terminal_block(stage: dict, base_url: str, payload: dict) -> str:
    return f"""cd {WORKDIR_CMD}
source .venv/bin/activate
pip install -r requirements.txt
{stage["serve"]}

# In another terminal — test this stage:
{render_curl(base_url, payload)}"""


st.set_page_config(page_title="Week 1 /ask Demo", layout="wide")
st.title("Week 1 — `/ask` Demo Runner")
st.caption("One page, five sections. Copy the commands below, start the matching server, then hit **Run test**.")

base_url = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")

st.sidebar.markdown("### Run this page")
st.sidebar.code(
    f"cd {WORKDIR_CMD}\nsource .venv/bin/activate\nstreamlit run demo_page.py",
    language="bash",
)

tabs = st.tabs([f"Demo {s['num']}: {s['title']}" for s in STAGES])

for tab, stage in zip(tabs, STAGES):
    with tab:
        st.subheader(f"Demo {stage['num']} — {stage['title']}")
        st.markdown(f"**Look for:** {stage['look_for']}")

        default_q = stage["dummy_question"]
        stage_question = st.text_input(
            "Question",
            default_q,
            key=f"q_{stage['num']}",
            placeholder="Type a question to send to /ask…",
        )

        force_bad = stage.get("dummy_force_bad", False)
        model = stage.get("dummy_model")
        if "force_bad" in stage["fields"]:
            force_bad = st.checkbox(
                "force_bad (break schema on attempt 1)",
                value=stage.get("dummy_force_bad", False),
                key=f"bad_{stage['num']}",
            )
        if "model" in stage["fields"]:
            options = [None, "gpt-4o", "gpt-4o-mini", "o3-mini"]
            default_model = stage.get("dummy_model")
            model = st.selectbox(
                "model",
                options,
                index=options.index(default_model) if default_model in options else 0,
                format_func=lambda m: m or "gpt-4o (default)",
                key=f"model_{stage['num']}",
            )

        payload = build_payload(stage_question, stage, force_bad, model)

        st.markdown("**Copy & run (terminal 1 — server, terminal 2 — curl):**")
        st.code(render_terminal_block(stage, base_url, payload), language="bash")

        if st.button("Run test", key=f"run_{stage['num']}", type="primary"):
            with st.spinner("Calling /ask..."):
                status, data = call_ask(base_url, payload)
            if status:
                st.markdown(f"**HTTP {status}**")
            st.json(data)

st.sidebar.divider()
st.sidebar.markdown("**Full reference:** `main.py` / `serve_stage5.py`")
