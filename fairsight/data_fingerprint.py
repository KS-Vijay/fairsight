# === data_fingerprint.py =========================================
"""
Data Fingerprinting Engine
-------------------------------------------------
Light-weight utility for detecting dataset overlap or leakage:
1. Generates SHA-256 hashes for each record (row-level fingerprinting)
2. Optionally projects high-dimensional feature vectors into an
   embedding space (Sentence-Transformers) and stores them in FAISS
   for approximate-nearest-neighbour (ANN) search.

Intended to be called from FSAuditor **before** bias / fairness
analysis so that any duplicated or previously-audited data can be
flagged early in the pipeline.
"""
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

__all__ = ["DataFingerprintEngine", "DuplicateRecord"]


class DuplicateRecord(Dict[str, Any]):
    """
    Convenience alias for returning duplicate-detection metadata.
    Keys:
        - index_a / index_b   : indices of the matching rows
        - distance            : cosine distance (embedding mode)
        - hash_match          : True if exact hash collision
    """
    pass


class DataFingerprintEngine:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        ann_metric: str = "cosine",
    ):
        self.use_embeddings = use_embeddings
        self._hashes: List[str] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._embeddings: Optional[np.ndarray] = None
        if self.use_embeddings:
            self._encoder = SentenceTransformer(model_name)
            self._metric = ann_metric
        else:
            self._encoder = None

    # ------------------------------------------------------------------ #
    #  Hash-based fingerprinting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_hash(row: pd.Series) -> str:
        """
        Stable SHA-256 hash of a DataFrame row after JSON serialisation.
        """
        json_bytes = json.dumps(row.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(json_bytes).hexdigest()

    def _build_hash_index(self, df: pd.DataFrame) -> None:
        self._hashes = [self._row_hash(row) for _, row in df.iterrows()]

    def _hash_lookup(self, row: pd.Series) -> Optional[int]:
        """
        Return index of the first colliding hash, or None.
        """
        h = self._row_hash(row)
        try:
            return self._hashes.index(h)
        except ValueError:
            return None

    # ------------------------------------------------------------------ #
    #  Embedding-based fingerprinting
    # ------------------------------------------------------------------ #
    def _build_embedding_index(self, df: pd.DataFrame) -> None:
        # Simple string representation of rows
        corpus = df.astype(str).agg(" ".join, axis=1).tolist()
        embeddings = self._encoder.encode(corpus, show_progress_bar=True)
        self._embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        if self._metric == "cosine":
            faiss.normalize_L2(self._embeddings)

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)

    def _embedding_lookup(
        self, row: pd.Series, top_k: int = 1
    ) -> List[DuplicateRecord]:
        if self._index is None or self._embeddings is None:
            raise RuntimeError("Embedding index is not built.")

        vec = self._encoder.encode(" ".join(row.astype(str).tolist())).astype("float32")
        if self._metric == "cosine":
            faiss.normalize_L2(vec)
        vec = np.expand_dims(vec, axis=0)

        distances, indices = self._index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                DuplicateRecord(
                    index_a=int(idx),
                    index_b=None,  # placeholder for caller to fill
                    distance=float(1 - dist) if self._metric == "cosine" else float(dist),
                    hash_match=False,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> None:
        """
        Persist fingerprints of an existing / reference dataset.
        """
        self._build_hash_index(df)
        if self.use_embeddings:
            self._build_embedding_index(df)

    def flag_potential_duplicates(
        self,
        new_df: pd.DataFrame,
        distance_threshold: float = 0.05,
        top_k: int = 3,
    ) -> List[DuplicateRecord]:
        """
        Compare `new_df` against the reference fingerprints.

        Returns a list of DuplicateRecord objects.
        """
        duplicates: List[DuplicateRecord] = []
        for idx, row in tqdm(new_df.iterrows(), total=len(new_df), desc="Fingerprinting"):
            # Hash collision first (fast path)
            match_idx = self._hash_lookup(row)
            if match_idx is not None:
                duplicates.append(
                    DuplicateRecord(
                        index_a=match_idx,
                        index_b=int(idx),
                        distance=0.0,
                        hash_match=True,
                    )
                )
                continue

            # Embedding similarity
            if self.use_embeddings:
                candidates = self._embedding_lookup(row, top_k=top_k)
                for cand in candidates:
                    if cand["distance"] <= distance_threshold:
                        cand["index_b"] = int(idx)
                        duplicates.append(cand)
        return duplicates

    # ------------------------------------------------------------------ #
    #  Persistence helpers
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise fingerprints to disk (hashes + embeddings + faiss index).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # hashes
        with open(path / "hashes.json", "w") as fp:
            json.dump(self._hashes, fp)
        # embeddings
        if self.use_embeddings and self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)
            faiss.write_index(self._index, str(path / "faiss.index"))

    def load(self, path: Union[str, Path]) -> None:
        """
        Load a previously-saved fingerprint database.
        """
        path = Path(path)
        with open(path / "hashes.json", "r") as fp:
            self._hashes = json.load(fp)
        if (path / "embeddings.npy").exists():
            self._embeddings = np.load(path / "embeddings.npy")
            self._index = faiss.read_index(str(path / "faiss.index"))
            self.use_embeddings = True
        else:
            self.use_embeddings = False
