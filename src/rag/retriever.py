"""
AOPRetriever: retrieves relevant AOP Wiki entries for a given text chunk.

Retrieval strategy (two-stage, no external dependencies beyond stdlib):
  1. Exact stressor match: any AOP whose stressor names appear verbatim in
     the query text is prioritised (high precision).
  2. BM25 over search_text: fills remaining slots with semantically related
     AOPs based on term overlap with titles, event names, and abstracts.

Usage:
    from src.rag.retriever import AOPRetriever

    retriever = AOPRetriever("new_data/aop_rag_index.json")

    # Returns a ready-to-inject string block
    rag_block = retriever.augment(user_content)

Build the index first with:
    python src/rag/build_index.py
"""

import json
import math
import re
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Minimal BM25 (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25:
    """BM25 Okapi over a corpus of pre-tokenized documents."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(1, n)

        self._tf: List[dict] = []
        df: dict = {}
        for doc in corpus:
            tf: dict = {}
            for tok in doc:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)
            for tok in set(doc):
                df[tok] = df.get(tok, 0) + 1

        self._idf: dict = {
            tok: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        tf = self._tf[doc_idx]
        dl = sum(tf.values())
        s = 0.0
        for tok in query_tokens:
            idf = self._idf.get(tok)
            if idf is None:
                continue
            tf_val = tf.get(tok, 0)
            s += idf * tf_val * (self.k1 + 1) / (
                tf_val + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def top_k(self, query_tokens: List[str], k: int) -> List[int]:
        scored = [
            (i, self.score(query_tokens, i))
            for i in range(len(self._tf))
        ]
        scored.sort(key=lambda x: -x[1])
        return [i for i, s in scored[:k] if s > 0.0]


# ---------------------------------------------------------------------------
# AOPRetriever
# ---------------------------------------------------------------------------

_RAG_HEADER = (
    "Relevant AOP Wiki entries that may relate to the text below "
    "(use as background knowledge only):\n"
)

# Marker used by DatasetGenerator to delimit the chunk text in the prompt
_TEXT_MARKER = "Text:\n"


class AOPRetriever:
    """
    Loads a pre-built AOP RAG index and retrieves the most relevant entries
    for a given query text.

    Args:
        index_path: Path to aop_rag_index.json produced by build_index.py.
        top_k: Number of AOP snippets to retrieve per query (default 6).
    """

    def __init__(self, index_path: str, top_k: int = 6):
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RAG index not found: {path}\n"
                "Build it first with: python src/rag/build_index.py"
            )
        with open(path, encoding="utf-8") as f:
            self._index = json.load(f)

        self.top_k = top_k

        corpus = [_tokenize(entry["search_text"]) for entry in self._index]
        self._bm25 = _BM25(corpus)

        # Pre-lowercase stressor name lists for fast exact-match scanning
        self._stressor_names: List[List[str]] = [
            entry.get("stressor_names") or [] for entry in self._index
        ]

        print(f"[RAG] index loaded: {len(self._index)} AOPs", flush=True)

    def retrieve(self, query_text: str) -> List[str]:
        """
        Return up to self.top_k AOP snippet strings, ordered by relevance.
        Stressor-matched entries come first, BM25 fills the rest.
        """
        q_lower = query_text.lower()

        # Stage 1: exact stressor name match
        stressor_hits: List[int] = []
        for i, names in enumerate(self._stressor_names):
            for name in names:
                if name and name in q_lower:
                    stressor_hits.append(i)
                    break

        # Stage 2: BM25 (fetch extra candidates to allow deduplication)
        q_tokens = _tokenize(query_text)
        bm25_hits = self._bm25.top_k(q_tokens, k=self.top_k * 3)

        # Merge, stressor hits first, no duplicates
        seen: set = set()
        ranked: List[int] = []
        for i in stressor_hits:
            if i not in seen:
                ranked.append(i)
                seen.add(i)
        for i in bm25_hits:
            if i not in seen and len(ranked) < self.top_k:
                ranked.append(i)
                seen.add(i)

        return [self._index[i]["snippet"] for i in ranked[: self.top_k]]

    def augment(self, user_content: str) -> str:
        """
        Insert retrieved AOP entries into user_content, immediately before
        the 'Text:\\n' marker that separates the instruction from the chunk.

        If the marker is absent, the RAG block is appended at the end.
        """
        # Find where the chunk text starts
        pos = user_content.rfind(_TEXT_MARKER)
        chunk_text = user_content[pos + len(_TEXT_MARKER):] if pos != -1 else user_content

        snippets = self.retrieve(chunk_text)
        if not snippets:
            return user_content

        rag_block = _RAG_HEADER + "\n\n".join(snippets)

        if pos != -1:
            prefix = user_content[: pos]
            return f"{prefix}{rag_block}\n\n{_TEXT_MARKER}{chunk_text}"
        else:
            return f"{user_content}\n\n{rag_block}"
