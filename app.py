"""
MCET RAG Assistant — Streamlit UI
==================================
Wraps the existing HybridRetriever in a presentable web interface.
On fallback (vector search) results, optionally calls an LLM to compose
a natural answer from the retrieved chunks.
Targets Streamlit >= 1.32 (no deprecated st.experimental_* calls).
"""

import os
import json
import datetime
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="MCET Assistant",
    page_icon="🎓",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths (relative to this file, which lives alongside mcet_retriever.py)
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(_DIR, "MCET_chunks.json")
STRUCTURED_DATA_PATH = os.path.join(_DIR, "MCET_structured_data.json")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_retriever():
    """Instantiate HybridRetriever ONCE per server session."""
    from mcet_retriever import HybridRetriever
    return HybridRetriever()


@st.cache_data(show_spinner=False)
def load_structured_data() -> dict | None:
    """Load MCET_structured_data.json for the Browse Data tab."""
    try:
        with open(STRUCTURED_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def load_chunks() -> dict:
    """Load raw chunks for source-text lookups in the chat."""
    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_data_timestamp() -> str:
    """Return human-readable last-modified time of the chunks file."""
    try:
        ts = os.path.getmtime(CHUNKS_PATH)
        return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y, %I:%M %p")
    except OSError:
        return "unknown"


def get_groq_key() -> str | None:
    """Read Groq API key from st.secrets, return None if not configured."""
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
        if key and isinstance(key, str) and key.strip() and key.strip() != "your-api-key-here":
            return key.strip()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source-display helper (defined early so it's available during rendering)
# ---------------------------------------------------------------------------
def render_sources(sources: list[dict]):
    """Render a collapsible Sources expander below an assistant message."""
    if not sources:
        return
    with st.expander("📄 Sources", expanded=False):
        for s in sources:
            cid = s.get("chunk_id", "—")
            sim = s.get("similarity", "—")
            text = s.get("text", "")
            mq = s.get("matched_question", "")

            st.markdown(f"**Chunk:** `{cid}` · **Similarity:** `{sim}`")
            if mq:
                st.markdown(f"**Matched question:** _{mq}_")
            if text:
                st.code(text, language=None)
            st.divider()


# ---------------------------------------------------------------------------
# Retriever init (with user-friendly error handling)
# ---------------------------------------------------------------------------
_retriever_error: str | None = None
try:
    with st.spinner("Loading embedding model… (first run downloads ~90 MB)"):
        retriever = load_retriever()
except ImportError as e:
    _retriever_error = (
        f"**Missing dependency:** `{e.name}`\n\n"
        "Run `pip install -r requirements.txt` and restart the app."
    )
except Exception as e:
    _retriever_error = (
        f"**Failed to initialise retriever:**\n\n`{e}`\n\n"
        "If this is your first run, make sure you have an internet connection "
        "so the `all-MiniLM-L6-v2` embedding model can be downloaded."
    )

chunks_lookup = load_chunks()


# ---------------------------------------------------------------------------
# Helper: run a query through the retriever and return a chat-ready dict
# ---------------------------------------------------------------------------
def process_query(query: str) -> dict:
    """Thin wrapper around retriever.retrieve() that adds display-friendly fields."""
    result = retriever.retrieve(query)

    if result["path"] == "fast_path":
        chunk_id = result.get("chunk_id", "")
        source_text = chunks_lookup.get(chunk_id, "")
        is_ambiguous = chunk_id.startswith("conflict_note")

        return {
            "type": "fast_path",
            "answer": result["answer"],
            "similarity": result["similarity"],
            "matched_question": result["matched_question"],
            "chunk_id": chunk_id,
            "source_text": source_text,
            "confidence": result.get("confidence", "unspecified"),
            "is_ambiguous": is_ambiguous,
        }
    else:
        chunks_out = []
        for c in result.get("retrieved_chunks", []):
            cid = c["chunk_id"]
            chunks_out.append({
                "chunk_id": cid,
                "text": c["text"],
                "similarity": c["similarity"],
                "is_ambiguous": cid.startswith("conflict_note"),
            })
        return {
            "type": "fallback",
            "best_fast_path_score": result.get("best_fast_path_score", 0),
            "chunks": chunks_out,
        }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
api_key = get_groq_key()
key_available = api_key is not None

with st.sidebar:
    st.markdown("## 🎓 MCET Assistant")
    st.caption(
        "Ask about HoDs, courses, fees, chairman, eligibility, "
        "and more — powered by a local RAG pipeline."
    )

    st.divider()

    # LLM toggle
    if key_available:
        use_llm = st.toggle("Use LLM generation", value=True, key="llm_toggle")
    else:
        use_llm = st.toggle(
            "Use LLM generation",
            value=False,
            disabled=True,
            key="llm_toggle",
            help="No Groq API key configured. Add GROQ_API_KEY to .streamlit/secrets.toml to enable.",
        )

    st.divider()
    st.markdown("**Example questions**")

    example_queries = [
        "Who is the chairman of MCET?",
        "What are the B.Tech fees?",
        "What PG programs are offered?",
        "Who is the HOD of CSE?",
        "What is the eligibility for lateral entry?",
    ]

    for eq in example_queries:
        if st.button(eq, key=f"ex_{eq}", use_container_width=True):
            st.session_state["pending_query"] = eq

    st.divider()

    with st.expander("ℹ️ About this project"):
        st.markdown(
            "This is a **Retrieval-Augmented Generation (RAG)** assistant "
            "built as a college project for MCET.\n\n"
            "It uses a **hybrid retrieval** architecture: incoming queries are "
            "first matched against pre-embedded synthetic Q&A pairs (fast path). "
            "If no close match is found, it falls back to semantic vector search "
            "over raw knowledge chunks, optionally followed by LLM generation "
            "to compose a natural answer.\n\n"
            "Embeddings: `all-MiniLM-L6-v2` via sentence-transformers (local, free).\n\n"
            "Generation: Groq `llama-3.3-70b-versatile` (optional, requires API key)."
        )

    st.divider()
    st.caption(f"📅 Data last updated: {get_data_timestamp()}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if _retriever_error:
    st.error(_retriever_error, icon="🚨")
    st.stop()

tab_chat, tab_browse = st.tabs(["💬 Chat", "📊 Browse Data"])


# ===========================  TAB 1: CHAT  ================================
with tab_chat:
    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render existing messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-render source expanders for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                render_sources(msg["sources"])

    # Determine input: sidebar button click or text box
    user_input = st.chat_input("Ask about MCET…")
    pending = st.session_state.pop("pending_query", None)
    query = pending or user_input

    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = process_query(query)
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")
                    result = None

            if result is not None:
                sources_data = []

                if result["type"] == "fast_path":
                    # ── Fast path: show pre-written answer, no LLM call ──
                    st.markdown(result["answer"])
                    st.caption(f"⚡ Instant match (similarity: {result['similarity']})")

                    if result["is_ambiguous"]:
                        st.warning(
                            "⚠️ This data point has known ambiguity — see source note below.",
                            icon="⚠️",
                        )

                    sources_data.append({
                        "chunk_id": result["chunk_id"],
                        "text": result["source_text"],
                        "similarity": result["similarity"],
                        "matched_question": result["matched_question"],
                    })
                    render_sources(sources_data)

                    assistant_content = result["answer"]

                else:
                    # ── Fallback path: retrieved chunks ──
                    chunks = result["chunks"]
                    if not chunks:
                        st.info("No relevant information found in the knowledge base.")
                        assistant_content = "No relevant information found."
                    else:
                        # Build sources_data first (needed for both LLM and raw display)
                        for c in chunks:
                            sources_data.append({
                                "chunk_id": c["chunk_id"],
                                "text": c["text"],
                                "similarity": c["similarity"],
                            })

                        llm_used = False

                        # Attempt LLM generation if enabled and key available
                        if use_llm and api_key:
                            with st.spinner("Generating answer…"):
                                try:
                                    from llm_generator import generate_answer
                                    llm_result = generate_answer(
                                        query=query,
                                        retrieved_chunks=chunks,
                                        api_key=api_key,
                                    )
                                except Exception as e:
                                    llm_result = {
                                        "success": False,
                                        "error": f"Import/call error: {type(e).__name__}",
                                        "answer": "",
                                        "model_used": "",
                                    }

                            if llm_result["success"]:
                                st.markdown(llm_result["answer"])
                                st.caption(
                                    f"🦙 Generated via Groq (Llama) "
                                    f"(model: {llm_result['model_used']})"
                                )
                                assistant_content = llm_result["answer"]
                                llm_used = True
                            else:
                                # LLM failed — warn and fall through to raw display
                                st.warning(
                                    f"LLM generation failed: {llm_result['error']} "
                                    "— showing retrieved context directly.",
                                    icon="⚠️",
                                )

                        # Raw chunk display (primary if no LLM, or fallback if LLM failed)
                        if not llm_used:
                            st.markdown("📚 **Retrieved from knowledge base:**")
                            answer_parts = []
                            for c in chunks:
                                st.markdown(f"- {c['text']}")
                                answer_parts.append(c["text"])

                                if c["is_ambiguous"]:
                                    st.warning(
                                        "⚠️ This chunk contains a note about data ambiguity.",
                                        icon="⚠️",
                                    )

                            if not use_llm and not api_key:
                                st.caption(
                                    "LLM generation not configured — "
                                    "showing retrieved context directly."
                                )
                            elif not use_llm:
                                st.caption("LLM generation disabled — showing raw context.")

                            assistant_content = (
                                "📚 Retrieved from knowledge base:\n"
                                + "\n".join(f"- {t}" for t in answer_parts)
                            )

                        # Always show sources regardless of LLM path
                        render_sources(sources_data)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": assistant_content,
                    "sources": sources_data,
                })


# ========================  TAB 2: BROWSE DATA  ============================
with tab_browse:
    structured = load_structured_data()

    if structured is None:
        st.info(
            "Structured data file not found (`MCET_structured_data.json`). "
            "The Chat tab still works — this tab just provides a browsable view "
            "of the underlying dataset.",
            icon="ℹ️",
        )
    else:
        # — Leadership —
        st.subheader("🏛️ College Leadership")
        if "leadership" in structured:
            st.dataframe(
                pd.DataFrame(structured["leadership"]),
                width="stretch",
                hide_index=True,
            )

        # — Department Heads —
        st.subheader("👩‍🏫 Department Heads")
        if "department_heads" in structured:
            st.dataframe(
                pd.DataFrame(structured["department_heads"]),
                width="stretch",
                hide_index=True,
            )

        st.divider()

        # — Courses —
        st.subheader("📘 Programs Offered")
        courses = structured.get("courses", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Undergraduate (B.E. / B.Tech)**")
            if "undergraduate" in courses:
                st.dataframe(
                    pd.DataFrame(courses["undergraduate"]),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("**Doctoral (Ph.D.)**")
            if "doctoral" in courses:
                st.dataframe(
                    pd.DataFrame(courses["doctoral"]),
                    width="stretch",
                    hide_index=True,
                )

        with col2:
            st.markdown("**Postgraduate (M.E. / MCA)**")
            if "postgraduate" in courses:
                st.dataframe(
                    pd.DataFrame(courses["postgraduate"]),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("**Fees & Eligibility**")
            if "fees" in structured:
                st.dataframe(
                    pd.DataFrame(structured["fees"]),
                    width="stretch",
                    hide_index=True,
                )
