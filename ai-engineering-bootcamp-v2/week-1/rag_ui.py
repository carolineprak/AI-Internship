"""Minimal Streamlit UI for the live Session 2 RAG API.

Source of truth = FastAPI on Render (or local). This page only calls:
  POST /ingest
  POST /ask

Run (from week-1):
  streamlit run rag_ui.py
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

DEFAULT_API_URL = (
    os.getenv("RAG_API_URL")
    or os.getenv("API_BASE_URL")
    or "https://ai-internship-bnrf.onrender.com"
).rstrip("/")

REFUSAL_PHRASE = "I don't have enough information to answer that."


def call_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict | str]:
    try:
        response = httpx.request(method, url, json=payload, timeout=180.0)
    except httpx.HTTPError as exc:
        return 0, f"Request failed: {exc}"
    try:
        body: dict | str = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


st.set_page_config(page_title="Northwind RAG UI", layout="centered")
st.title("Northwind RAG")
st.caption("Thin client for live FastAPI `/ingest` + `/ask`. No local RAG.")

with st.sidebar:
    st.header("API")
    api_url = st.text_input(
        "Base URL",
        value=DEFAULT_API_URL,
        help="Override with env RAG_API_URL or API_BASE_URL. No secrets needed here.",
    ).rstrip("/")
    st.caption(f"Using: `{api_url}`")

tab_ingest, tab_ask = st.tabs(["Ingest", "Ask"])

with tab_ingest:
    st.subheader("POST /ingest")
    document_id = st.text_input("document_id", value="handbook", key="ingest_doc_id")
    source = st.text_input("source (optional)", value="", key="ingest_source")
    text = st.text_area("Document text", height=240, key="ingest_text")
    if st.button("Ingest", type="primary", key="ingest_btn"):
        if not document_id.strip() or not text.strip():
            st.error("document_id and text are required.")
        else:
            payload = {
                "text": text.strip(),
                "document_id": document_id.strip(),
            }
            if source.strip():
                payload["source"] = source.strip()
            status, body = call_json("POST", f"{api_url}/ingest", payload)
            st.write(f"HTTP {status}")
            if status == 200 and isinstance(body, dict):
                st.success(
                    f"Indexed **{body.get('chunks_indexed')}** chunks "
                    f"for `{body.get('document_id')}` ({body.get('status')})"
                )
            else:
                st.error("Ingest failed")
            st.code(json.dumps(body, indent=2), language="json")

with tab_ask:
    st.subheader("POST /ask")
    question = st.text_area(
        "Question",
        value="What is the remote work policy?",
        height=100,
        key="ask_question",
    )
    top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
    if st.button("Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.error("question is required.")
        else:
            status, body = call_json(
                "POST",
                f"{api_url}/ask",
                {"question": question.strip(), "top_k": top_k},
            )
            st.write(f"HTTP {status}")
            if status != 200 or not isinstance(body, dict):
                st.error("Ask failed")
                st.code(json.dumps(body, indent=2) if not isinstance(body, str) else body)
            else:
                answer_obj = body.get("answer") or {}
                answer_text = str(answer_obj.get("answer") or "")
                chunk_ids = body.get("retrieved_chunk_ids") or []
                is_refusal = REFUSAL_PHRASE.lower() in answer_text.lower()

                if is_refusal:
                    st.warning("Refusal — not enough information in retrieved context.")
                else:
                    st.success("Answered from retrieved context")

                st.markdown("### Answer")
                st.write(answer_text)

                cols = st.columns(3)
                cols[0].metric("confidence", answer_obj.get("confidence"))
                cols[1].metric("tokens_used", body.get("tokens_used"))
                cols[2].metric("cost_usd", body.get("cost_usd"))

                st.markdown("### Citations / retrieved chunks")
                if chunk_ids:
                    for cid in chunk_ids:
                        st.code(cid, language=None)
                else:
                    st.info("No retrieved_chunk_ids in response.")

                # Highlight document_id mentions in the answer text when present
                if "document_id" in answer_text.lower() or "doc" in answer_text.lower():
                    st.caption("If the model cited a document_id in the answer text, it appears above.")

                with st.expander("Full JSON response"):
                    st.code(json.dumps(body, indent=2), language="json")
