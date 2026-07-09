"""
Verification Script for ChromaDB Hybrid Retriever
=================================================
Tests that the new persistent database client works and matches schema requirements.
"""

import unittest
import os
import shutil

from mcet_retriever import HybridRetriever


class TestChromaDBRetriever(unittest.TestCase):

    def setUp(self):
        # We can test with the active database, or create/reset collections.
        # For simplicity, we just initialize the retriever.
        self.retriever = HybridRetriever()

    def test_database_initialization(self):
        """Test that collections are created and populated."""
        self.assertIsNotNone(self.retriever.client)
        self.assertIsNotNone(self.retriever.qa_collection)
        self.assertIsNotNone(self.retriever.chunk_collection)
        
        # Verify that collection items are populated
        self.assertGreater(self.retriever.qa_collection.count(), 0)
        self.assertGreater(self.retriever.chunk_collection.count(), 0)

    def test_fast_path_retrieval(self):
        """Test that matching questions resolve via the fast-path."""
        query = "Who is the HOD of CSE?"
        result = self.retriever.retrieve(query, force_refresh=True)
        
        # Validate fast path structure
        self.assertEqual(result["path"], "fast_path")
        self.assertIn("answer", result)
        self.assertIn("matched_question", result)
        self.assertIn("similarity", result)
        self.assertGreaterEqual(result["similarity"], 0.72)
        self.assertIn("Dr. D. Sivaganesan", result["answer"])

    def test_fallback_retrieval(self):
        """Test that non-matching questions fall back to vector search."""
        query = "What documents do I need to submit for lateral entry admission?"
        result = self.retriever.retrieve(query, force_refresh=True)
        
        # Validate fallback structure
        self.assertEqual(result["path"], "fallback_vector_search")
        self.assertIn("retrieved_chunks", result)
        self.assertGreater(len(result["retrieved_chunks"]), 0)
        
        # Check first chunk structure
        first_chunk = result["retrieved_chunks"][0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("similarity", first_chunk)


if __name__ == "__main__":
    unittest.main()
