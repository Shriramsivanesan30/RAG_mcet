"""
Answer Validator for MCET RAG Assistant
========================================
Validates LLM consistency across queries and generates standardized correction messages.
"""

class AnswerValidator:
    # Persistent class-level dictionary to store query history across instantiations
    _history = {}

    def __init__(self):
        self.answer_history = AnswerValidator._history
    
    def validate_consistency(self, query: str, new_answer: str, context: str) -> dict:
        """
        Check if new answer conflicts with previous answers.
        Returns: { "consistent": bool, "correction": str or None }
        """
        if not query or not new_answer:
            return {"consistent": True, "correction": None}

        query_hash = hash(query.lower().strip())
        
        if query_hash in self.answer_history:
            previous = self.answer_history[query_hash]
            
            # If the answer is different (and not just exact string match)
            if previous.strip() != new_answer.strip():
                correction_msg = (
                    f"Correction: My previous response was incomplete/incorrect. "
                    f"Based on the full context, the correct information is: {new_answer} "
                    f"I apologize for the earlier oversight."
                )
                return {
                    "consistent": False,
                    "correction": correction_msg
                }
        
        # Store for future reference
        self.answer_history[query_hash] = new_answer
        return {"consistent": True, "correction": None}
