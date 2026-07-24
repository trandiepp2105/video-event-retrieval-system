from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .io_utils import load_pickle, save_json, save_pickle
from .metadata import MetadataStore
from .schemas import FrameRangeMetadata, SearchResult


def simple_tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", str(text).lower(), flags=re.UNICODE) if token]


@dataclass
class OCRDocument:
    index_id: int
    video_id: str
    frame_id: int
    text: str


class BM25Index:
    def __init__(
        self,
        *,
        index_name: str,
        documents: list[OCRDocument],
        doc_token_freqs: list[dict[str, int]],
        doc_lengths: list[int],
        term_doc_freq: dict[str, int],
        avg_doc_len: float,
        metadata_store: MetadataStore,
        config: dict,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index_name = str(index_name)
        self.documents = documents
        self.doc_token_freqs = doc_token_freqs
        self.doc_lengths = doc_lengths
        self.term_doc_freq = term_doc_freq
        self.avg_doc_len = float(avg_doc_len)
        self.metadata_store = metadata_store
        self.config = dict(config)
        self.k1 = float(k1)
        self.b = float(b)

    @classmethod
    def build(
        cls,
        *,
        documents: list[OCRDocument],
        metadata_store: MetadataStore,
        index_name: str,
        config: dict,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        doc_token_freqs: list[dict[str, int]] = []
        doc_lengths: list[int] = []
        term_doc_freq: dict[str, int] = {}

        for doc in documents:
            tokens = simple_tokenize(doc.text)
            token_freq: dict[str, int] = {}
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            doc_token_freqs.append(token_freq)
            doc_lengths.append(len(tokens))
            for token in token_freq.keys():
                term_doc_freq[token] = term_doc_freq.get(token, 0) + 1

        avg_doc_len = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
        return cls(
            index_name=index_name,
            documents=documents,
            doc_token_freqs=doc_token_freqs,
            doc_lengths=doc_lengths,
            term_doc_freq=term_doc_freq,
            avg_doc_len=avg_doc_len,
            metadata_store=metadata_store,
            config=config,
            k1=k1,
            b=b,
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(
            {
                "documents": self.documents,
                "doc_token_freqs": self.doc_token_freqs,
                "doc_lengths": self.doc_lengths,
                "term_doc_freq": self.term_doc_freq,
                "avg_doc_len": self.avg_doc_len,
                "k1": self.k1,
                "b": self.b,
            },
            output_dir / "bm25.pkl",
        )
        self.metadata_store.save(output_dir / "metadata.json")
        save_json(self.config, output_dir / "config.json")

    @classmethod
    def load(cls, input_dir: Path) -> "BM25Index":
        payload = load_pickle(input_dir / "bm25.pkl")
        metadata_store = MetadataStore.load(input_dir / "metadata.json")
        from .io_utils import load_json

        config = load_json(input_dir / "config.json")
        return cls(
            index_name=config.get("index_name", input_dir.name),
            documents=payload["documents"],
            doc_token_freqs=payload["doc_token_freqs"],
            doc_lengths=payload["doc_lengths"],
            term_doc_freq=payload["term_doc_freq"],
            avg_doc_len=float(payload["avg_doc_len"]),
            metadata_store=metadata_store,
            config=config,
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )

    def _idf(self, token: str, num_docs: int) -> float:
        doc_freq = self.term_doc_freq.get(token, 0)
        if doc_freq <= 0:
            return 0.0
        return math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []
        num_docs = len(self.documents)
        scores: list[tuple[int, float]] = []
        avg_doc_len = self.avg_doc_len if self.avg_doc_len > 0 else 1.0

        for row, doc in enumerate(self.documents):
            score = 0.0
            token_freq = self.doc_token_freqs[row]
            doc_len = self.doc_lengths[row]
            for token in query_tokens:
                tf = token_freq.get(token, 0)
                if tf <= 0:
                    continue
                idf = self._idf(token, num_docs)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
                score += idf * (numerator / denominator)
            if score > 0:
                scores.append((row, float(score)))

        scores.sort(key=lambda item: item[1], reverse=True)
        results: list[SearchResult] = []
        for row, score in scores[: int(top_k)]:
            meta = self.metadata_store.get(self.documents[row].index_id)
            results.append(
                SearchResult(
                    index_id=meta.index_id,
                    video_id=meta.video_id,
                    frame_start=meta.frame_start,
                    frame_end=meta.frame_end,
                    score=score,
                    item_id=meta.item_id,
                )
            )
        return results
