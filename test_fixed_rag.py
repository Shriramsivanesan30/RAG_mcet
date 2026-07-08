"""
Unit and Integration Tests for RAG Application-Level Fixes
===========================================================
"""

import unittest
from unittest.mock import patch, MagicMock

from citation_formatter import format_citations
from ui_cleaner import clean_ui_output
from answer_validator import AnswerValidator
from llm_generator import generate_answer, clean_input_text


class TestRAGApplicationFixes(unittest.TestCase):

    def setUp(self):
        # Reset AnswerValidator history before each test
        AnswerValidator._history.clear()

    def test_ui_cleaner(self):
        """Test that system/UI artifacts are successfully stripped."""
        raw_output = (
            "The head of the department is Dr. L. Meenachi.\n"
            "Activate Windows\n"
            "Go to Settings to activate Windows.\n"
            "[Sources](#)\n"
            "Instant match (similarity: 0.95)"
        )
        cleaned = clean_ui_output(raw_output)
        self.assertNotIn("Activate Windows", cleaned)
        self.assertNotIn("Go to Settings to activate Windows.", cleaned)
        self.assertNotIn("[Sources](#)", cleaned)
        self.assertNotIn("Instant match (similarity: 0.95)", cleaned)
        self.assertIn("Dr. L. Meenachi", cleaned)

    def test_citation_formatter(self):
        """Test that citations are formatted and appended correctly."""
        response = "The head of the IT department is Dr. L. Meenachi [hod_it_001]."
        chunks = [
            {
                "chunk_id": "hod_it_001",
                "text": "Head of the Department, Information Technology (IT): Dr. L. Meenachi.",
                "similarity": 0.8932
            }
        ]
        formatted = format_citations(response, chunks)
        self.assertIn("[Source: hod_it_001]", formatted)
        self.assertIn("Similarity: 0.8932", formatted)
        self.assertIn("Head of the Department", formatted)

    def test_answer_validator_consistency(self):
        """Test that consecutive inconsistent answers trigger corrections."""
        validator = AnswerValidator()
        query = "Who is the HOD of IT?"
        
        # Turn 1: Assistant doesn't know
        ans1 = "I couldn't find that information in the provided context."
        res1 = validator.validate_consistency(query, ans1, "")
        self.assertTrue(res1["consistent"])
        self.assertIsNone(res1["correction"])

        # Turn 2: Assistant now finds it
        ans2 = "Dr. L. Meenachi is the Head of the Department."
        res2 = validator.validate_consistency(query, ans2, "")
        self.assertFalse(res2["consistent"])
        self.assertIsNotNone(res2["correction"])
        self.assertIn("Correction: My previous response was incomplete/incorrect.", res2["correction"])
        self.assertIn("Dr. L. Meenachi", res2["correction"])

    @patch("llm_generator.Groq")
    def test_generate_answer_mocked(self, mock_groq):
        """Test the end-to-end generate_answer flow with mocked Groq client."""
        # Setup mock Groq response
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        
        mock_choice = MagicMock()
        mock_choice.message.content = "Dr. L. Meenachi is the HOD [hod_it_001].\nActivate Windows"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        context = "[hod_it_001] Head of the Department, Information Technology (IT): Dr. L. Meenachi."
        query = "Who is the head of IT?"
        
        # Test signature 2: generate_answer(query, context) -> str
        result = generate_answer(query, context, api_key="mock-key")
        
        # Assertions
        self.assertIsInstance(result, str)
        self.assertNotIn("Activate Windows", result)
        self.assertIn("Dr. L. Meenachi", result)
        self.assertIn("[Source: hod_it_001]", result)


if __name__ == "__main__":
    unittest.main()
