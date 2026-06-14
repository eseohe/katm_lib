"""Document builder module for KATM."""

from typing import List

import nltk

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class DocumentBuilder:
    """Builds documents from raw texts using various strategies."""

    def __init__(self, strategy: str = "paragraph_group", chunk_size: int = 5):
        """Initialize DocumentBuilder.

        Args:
            strategy: One of "paragraph_group", "fixed_window", or "post_group".
            chunk_size: Number of units to group into each document.

        Raises:
            ValueError: If strategy is not recognized.
        """
        valid_strategies = {"paragraph_group", "fixed_window", "post_group"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {valid_strategies}")

        self.strategy = strategy
        self.chunk_size = chunk_size

    def build(self, raw_texts: List[str]) -> List[str]:
        """Build documents from raw texts.

        Args:
            raw_texts: List of raw text strings.

        Returns:
            List of document strings.
        """
        if not raw_texts:
            return []

        if self.strategy == "paragraph_group":
            return self._build_paragraph_group(raw_texts)
        elif self.strategy == "fixed_window":
            return self._build_fixed_window(raw_texts)
        elif self.strategy == "post_group":
            return self._build_post_group(raw_texts)

        return []

    def _build_paragraph_group(self, raw_texts: List[str]) -> List[str]:
        """Split on double newlines, group into chunks."""
        all_paragraphs = []
        for text in raw_texts:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            all_paragraphs.extend(paragraphs)

        return self._group_chunks(all_paragraphs)

    def _build_fixed_window(self, raw_texts: List[str]) -> List[str]:
        """Split into sentences, group into chunks of chunk_size sentences."""
        all_sentences = []
        for text in raw_texts:
            sentences = nltk.sent_tokenize(text)
            all_sentences.extend([s.strip() for s in sentences if s.strip()])

        return self._group_chunks(all_sentences)

    def _build_post_group(self, raw_texts: List[str]) -> List[str]:
        """Each element in raw_texts is a post; group chunk_size posts into one doc."""
        return self._group_chunks(list(raw_texts))

    def _group_chunks(self, items: List[str]) -> List[str]:
        """Group items into chunks of chunk_size."""
        if not items:
            return []

        chunks = []
        for i in range(0, len(items), self.chunk_size):
            chunk = items[i : i + self.chunk_size]
            chunks.append(" ".join(chunk))

        return chunks