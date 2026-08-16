"""
Minimal Streamlit UI for the Northwind ADK RAG agent.

Run (from this folder, venv active):
  streamlit run northwind_streamlit.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from northwind_rag_agent import (  # noqa: E402
    MAX_AGENT_STEPS,
    MODEL,
    RAG_API_URL,
    run_agent,
)


def run_async(coro):
    """Streamlit-safe asyncio runner."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


st.set_page_config(page_title="Northwind Policy Agent", layout="centered")
st.title("Northwind Policy Agent")
st.caption("Google ADK · real tool: search_docs → Session 2 `/debug/retrieve`")

with st.sidebar:
    st.header("Config")
    st.write(f"**RAG API:** `{RAG_API_URL}`")
    st.write(f"**Model:** `{MODEL}`")
    st.write(f"**Max steps:** `{MAX_AGENT_STEPS}`")
    st.caption("Set RAG_API_URL / GEMINI_MODEL / GOOGLE_API_KEY in `.env`.")

question = st.text_area(
    "Policy question",
    value="What is the remote work policy?",
    height=100,
)

col1, col2 = st.columns(2)
with col1:
    run_clicked = st.button("Run agent", type="primary")
with col2:
    refuse_demo = st.button("Try refusal question")

if refuse_demo:
    question = "What is the CEO's favorite pizza topping?"
    st.session_state["last_q"] = question

if run_clicked or refuse_demo:
    q = (st.session_state.get("last_q") if refuse_demo else question) or question
    with st.spinner("Agent running (Think → Act → Observe)..."):
        result = run_async(run_agent(q))

    if result.get("error"):
        st.error(result["error"])
    else:
        st.markdown("### Final answer")
        st.write(result.get("final_answer") or "")

        st.markdown("### Think → Act → Observe")
        steps = result.get("steps") or []
        if not steps:
            st.info("No step events captured.")
        for i, step in enumerate(steps, start=1):
            phase = step.get("phase", "?")
            detail = step.get("detail", "")
            if phase == "Act":
                st.success(f"{i}. **Act** — {detail}")
            elif phase == "Observe":
                st.warning(f"{i}. **Observe** — {detail}")
            else:
                st.info(f"{i}. **Think** — {detail}")

        with st.expander("Full JSON"):
            st.code(json.dumps(result, indent=2), language="json")
