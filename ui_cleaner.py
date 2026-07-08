"""
UI Cleaner for MCET RAG Assistant
=================================
Removes system and UI artifacts from the final output response.
"""

def clean_ui_output(raw_response: str) -> str:
    """
    Remove system/UI artifacts from the final output.
    """
    if not raw_response:
        return ""

    artifacts = [
        "Activate Windows",
        "Go to Settings to activate Windows.",
        "[Sources](#)",  # Remove placeholder links
        "Instant match (similarity:",
        "similarity:",
        "Sources",
    ]
    
    cleaned = raw_response
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    
    # Clean up extra blank lines
    cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
    
    return cleaned.strip()
