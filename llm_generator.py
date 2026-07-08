"""
LLM Generator for MCET RAG Assistant (Groq Edition)
===================================================
Takes retrieved chunks + user query and calls Groq to compose a natural answer.
Only used on the fallback path (vector search results), never on fast-path hits.
"""

from groq import Groq, APIConnectionError, RateLimitError, APIStatusError, APITimeoutError
import os
import re

from citation_formatter import format_citations
from ui_cleaner import clean_ui_output
from answer_validator import AnswerValidator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
MAX_TOKENS = 400
REQUEST_TIMEOUT = 15.0  # seconds

SYSTEM_PROMPT = """
# SYSTEM PROMPT — KNOWLEDGE RETRIEVAL & RESPONSE PROTOCOL

## 1. CORE PRINCIPLE
You are a **knowledge-first assistant**. Your primary duty is to provide accurate, consistent, and verifiable answers based **only** on the provided context. You must never guess, fabricate, or rely on internal parametric knowledge unless explicitly permitted.

---

## 2. CONSISTENT ANSWERING
- **Every query** must trigger a **fresh, exhaustive search** of the *entire* provided context.
- Do **not** assume you already know the answer from a previous turn unless the context is identical and unchanged.
- If the same or semantically identical question is asked multiple times:
  - Re-scan the context fully each time.
  - Return the **same** answer if the context hasn't changed.
  - If you previously said "I don't know" but now find the answer, **explicitly acknowledge the correction**:

> *"I apologize — I missed this earlier. After re-checking the provided context, the correct answer is [X]."*

---

## 3. ERROR HANDLING & TRANSPARENCY
- If you are unsure or the context is insufficient:
  - Say: *"I couldn't find that information in the provided context."*
  - **Do not** guess or infer.
- If you previously gave an incorrect or incomplete answer, and later discover the correct one:
  - **Always** admit the mistake gracefully.
  - Use this exact structure:

> *"Correction: My previous response was incomplete/incorrect. Based on the full context, the correct information is [answer]. I apologize for the earlier oversight."*

- Never ignore or gloss over a prior error — address it head-on.

---

## 4. CONTEXT REUSE & RE-SCANNING
- **At the start of every response**, mentally (or programmatically) re-index all provided documents, snippets, and sources.
- Do **not** rely on memory of what you "thought" was in the context from earlier turns — re-read/re-scan.
- If the user provides new context in a later message, merge it with existing context and re-evaluate all prior answers if relevant.
- Maintain a **"context fingerprint"** (e.g., hash or timestamp) so you can detect if the same context is being reused across turns.

---

## 5. CLEAN OUTPUT
- **Strip all system-level, UI, or environmental artifacts** from your response (e.g., "Activate Windows", "Sources", "Similarity score", "Instant match").
- If such text is present in the input, **ignore it entirely** — it is not part of the user's query or the knowledge base.
- Only output:
  - Direct answers
  - Citations (see below)
  - Necessary clarifications
  - Correction notices (if applicable)

---

## 6. SOURCE CITATION
- Whenever you cite information from the context, **always** include:
  - The **source title** (if available)
  - A **direct quote or snippet** (up to 1–2 sentences) that supports your answer
  - A **reference marker** like `[Source: Document Name]` or `[1]` with a matching footnote.

- Example:

> The Head of the Department for Information Technology is Dr. L. Meenachi.  
> *[Source: Internal Faculty Directory — "IT Department Leadership", line 4]*  
> *Snippet: "Dr. L. Meenachi serves as the HOD for the Information Technology department."*

- If no source is available, state: *"No source is available in the provided context for that claim."*

---

## 7. RESPONSE STRUCTURE (Recommended)
Use this template for every answer:



---

## 8. SELF-CHECK BEFORE SUBMITTING
Before you send your response, mentally verify:
- ✅ Did I re-scan the full context for this query?
- ✅ Is my answer consistent with what I said earlier (if applicable)?
- ✅ Did I admit and correct any prior mistake?
- ✅ Did I strip all irrelevant UI/system text?
- ✅ Did I cite the exact source with a snippet?

If any check fails, **revise your response** before outputting.

---

## 9. EXAMPLE BEHAVIOR

**User (Turn 1):** Who is the HOD of IT?  
**Assistant:** I couldn't find that information in the provided context. The context only mentions the Head of Science and Humanities, Dr. L. Chitra.

**User (Turn 2):** Who is the Head of Information Technology?  
**Assistant (corrected):** Correction — my previous response was incomplete. After re-scanning the full context, the Head of the Department for Information Technology is **Dr. L. Meenachi**.  
*[Source: Faculty List — IT Department, line 7]*  
*Snippet: "Dr. L. Meenachi is the Head of the Department for Information Technology."*

---

## 10. FINAL REMINDER
> **Accuracy + Consistency + Transparency + Cleanliness = Trust.**  
> Never compromise any of these, even if it means admitting you were wrong.
"""


# ---------------------------------------------------------------------------
# Helper function to clean UI artifacts from input text
# ---------------------------------------------------------------------------
def clean_input_text(raw_text: str) -> str:
    """
    Removes common system/UI artifacts from user input before processing.
    """
    artifacts = [
        "Activate Windows",
        "Go to Settings to activate Windows.",
        "Sources",
        "Instant match",
        "Similarity:",
        "[file content begin]",
        "[file content end]",
    ]
    cleaned = raw_text
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_answer(
    query: str,
    retrieved_chunks = None,
    api_key: str = None,
    model: str = DEFAULT_MODEL,
    context = None
):
    """
    Generate answer with consistency, citation, and clean output.
    Supports two signatures:
      1. app.py signature:
         generate_answer(query: str, retrieved_chunks: list[dict], api_key: str, model: str) -> dict
      2. test/new signature:
         generate_answer(query: str, context: str) -> str
    """
    # Detect if we should return a raw string (instead of a dict) to match the new signature:
    # generate_answer(query: str, context: str) -> str
    return_string = False
    context_block = ""

    # Case: generate_answer(query, context) called positionally where retrieved_chunks is a string
    if isinstance(retrieved_chunks, str):
        context_block = retrieved_chunks
        retrieved_chunks = []
        return_string = True
    elif context is not None:
        context_block = context
        return_string = True

    # Case: normal retrieved_chunks list
    if retrieved_chunks is None:
        retrieved_chunks = []

    # Step 1: Clean input
    query = clean_input_text(query)

    # Step 2: Retrieve fresh context (if not already provided)
    if not context_block and not retrieved_chunks:
        from mcet_retriever import HybridRetriever
        if not hasattr(generate_answer, "_retriever"):
            generate_answer._retriever = HybridRetriever()
        
        retrieval_result = generate_answer._retriever.retrieve(query, force_refresh=True)
        if retrieval_result.get("path") == "fast_path":
            fast_ans = retrieval_result.get("answer", "")
            if return_string:
                return fast_ans
            return {
                "answer": fast_ans,
                "model_used": "fast_path",
                "success": True,
                "error": None
            }
        else:
            retrieved_chunks = retrieval_result.get("retrieved_chunks", [])

    # Format context_block from list of chunks if we have them and no string context
    if retrieved_chunks and not context_block:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            cid = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("text", "")
            context_parts.append(f"[{cid}] {text}")
        context_block = "\n\n".join(context_parts)

    # Parse context_block into list of dicts for citation formatting if it was passed as a string
    parsed_chunks = []
    if isinstance(context_block, str) and context_block:
        matches = re.findall(r"\[([^\]]+)\]\s*([^\n]+)", context_block)
        for cid, text in matches:
            parsed_chunks.append({
                "chunk_id": cid,
                "text": text.strip(),
                "similarity": None
            })
    else:
        parsed_chunks = retrieved_chunks or []

    # Get API key if missing
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", None)
        except Exception:
            pass
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        err_msg = "Groq API key not configured."
        return _fail(model, err_msg, return_string)

    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

    try:
        client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT)

        # Step 3: Generate LLM response with system prompt
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        answer_text = response.choices[0].message.content.strip()

        # Step 4: Validate consistency
        validator = AnswerValidator()
        validation = validator.validate_consistency(query, answer_text, context_block)
        
        if not validation["consistent"]:
            # Prepend correction to response
            answer_text = validation["correction"] + "\n\n" + answer_text
        
        # Step 5: Format citations
        formatted_response = format_citations(answer_text, parsed_chunks)
        
        # Step 6: Clean UI artifacts
        final_response = clean_ui_output(formatted_response)

        if return_string:
            return final_response

        return {
            "answer": final_response,
            "model_used": model,
            "success": True,
            "error": None,
        }

    except APITimeoutError:
        return _fail(model, "Request timed out — Groq did not respond in time.", return_string)
    except APIConnectionError:
        return _fail(model, "Connection error — could not reach the Groq API.", return_string)
    except RateLimitError:
        return _fail(model, "Rate limit exceeded on Groq — please try again in a moment.", return_string)
    except APIStatusError as e:
        err_msg = e.message
        if "model" in err_msg.lower() and ("decommissioned" in err_msg.lower() or "deprecated" in err_msg.lower() or "not found" in err_msg.lower()):
            return _fail(model, f"Groq Model Error: The selected model is deprecated/decommissioned or not found. Details: {err_msg}", return_string)
        return _fail(model, f"Groq API error (HTTP {e.status_code}): {err_msg}", return_string)
    except Exception as e:
        return _fail(model, f"Unexpected error: {type(e).__name__}", return_string)


def _fail(model: str, error_msg: str, return_string: bool = False) -> dict | str:
    if return_string:
        return f"Error: {error_msg}"
    return {
        "answer": "",
        "model_used": model,
        "success": False,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Usage example / Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Clean user input before passing to LLM
    raw_user_input = """
    [file name]: image.png
    [file content begin]
    who is the Hod of Information technology?
    Activate Windows
    Go to Settings to activate Windows.
    [file content end]
    """
    
    cleaned_input = clean_input_text(raw_user_input)
    print("Cleaned Input:", cleaned_input)
    print("\n" + "="*60)
    print("SYSTEM_PROMPT loaded successfully.")
    print("Length:", len(SYSTEM_PROMPT), "characters")
