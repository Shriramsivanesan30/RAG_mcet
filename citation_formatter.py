"""
Citation Formatter for MCET RAG Assistant
==========================================
Formats retrieved chunk details and appends verified source citations to response.
"""

def format_citations(response: str, retrieved_chunks: list[dict]) -> str:
    """
    Format source citations and append/replace references in the response text.
    """
    if not response:
        return ""
    if not retrieved_chunks or not isinstance(retrieved_chunks, list):
        return response

    citations_to_add = []
    # Identify which chunks are referenced in the response text (by ID or 1-based index)
    for idx, chunk in enumerate(retrieved_chunks, 1):
        if not isinstance(chunk, dict):
            continue
        cid = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        sim = chunk.get("similarity", 0.0)
        
        # Check if the response references this chunk
        referenced = False
        if cid and cid in response:
            referenced = True
        elif f"[{idx}]" in response:
            referenced = True
        elif f"Source: {cid}" in response:
            referenced = True

        # Always include the top matching chunk, or all chunks that are referenced
        if referenced or idx == 1:
            snippet = text[:200] + "..." if len(text) > 200 else text
            citations_to_add.append({
                "id": cid,
                "text": text,
                "snippet": snippet,
                "similarity": sim,
                "index": idx
            })

    if not citations_to_add:
        return response

    # Format the citations section
    citation_lines = ["\n\n---", "### Sources & Citations"]
    for citation in citations_to_add:
        line = f"- **[Source: {citation['id']}]**"
        if citation['similarity']:
            line += f" (Similarity: {citation['similarity']:.4f})"
        line += f"\n  > \"{citation['snippet']}\""
        citation_lines.append(line)

    return response + "\n" + "\n".join(citation_lines)
