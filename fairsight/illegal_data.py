# === illegal_data.py ==============================================
"""
Illegal Data Detector (stand-alone utility)
-------------------------------------------------
Given arbitrary user-supplied data (currently image folder OR a
tabular DataFrame), checks whether any items are present in an
external *live* generative-model index or a known blacklist.

A typical use case is to prevent incorporation of copyrighted or
disallowed material (e.g. Studio Ghibli images) into training data.
"""
import hashlib
import io
import json
from pathlib import Path
from typing import List, Dict, Union, Tuple

import requests
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import pandas as pd

__all__ = ["IllegalDataDetector", "Violation"]


Violation = Dict[str, Union[str, float]]  # simple alias


class IllegalDataDetector:
    """
    A minimal API wrapper that:
      1. Generates embeddings for each incoming item
      2. Queries an external vector API   (e.g. CLIP-based similarity endpoint)
      3. Flags items whose similarity exceeds a configurable threshold
    """

    def __init__(
        self,
        remote_endpoint: str,
        model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        top_k: int = 5,
        similarity_threshold: float = 0.9,
    ):
        self.endpoint = remote_endpoint
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.threshold = similarity_threshold

    # ------------------------------------------------------------------ #
    #  Core helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((224, 224))
        pixels = np.asarray(img).astype(np.float32) / 255.0
        # flatten for sentence-transformers text path
        return self.encoder.encode(pixels.flatten())

    def _remote_similarity_search(self, vector: np.ndarray) -> List[Tuple[str, float]]:
        """
        POST the vector to `self.endpoint` and expect JSON:
           [{ "id": "some_id", "score": 0.97 }, ...]
        """
        payload = {"vector": vector.tolist(), "top_k": self.top_k}
        rsp = requests.post(self.endpoint, json=payload, timeout=30)
        rsp.raise_for_status()
        results = rsp.json()
        return [(item["id"], item["score"]) for item in results]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def scan_images(self, folder: Union[str, Path]) -> List[Violation]:
        """
        Iterate through all image files in `folder`, flagging violations.
        """
        folder = Path(folder)
        assert folder.is_dir(), f"{folder} is not a directory"
        flagged: List[Violation] = []

        for path in tqdm(list(folder.glob("*.*")), desc="Scanning images"):
            try:
                img = Image.open(path)
            except Exception:
                continue  # skip non-images

            vec = self._embed_image(img)
            matches = self._remote_similarity_search(vec)

            for ref_id, score in matches:
                if score >= self.threshold:
                    flagged.append(
                        {
                            "filepath": str(path),
                            "reference_id": ref_id,
                            "similarity": float(score),
                            "sha256": self._sha256_file(path),
                        }
                    )
                    break  # one hit is enough
        return flagged


def detect_illegal_data(df: pd.DataFrame, *args, **kwargs):
    """
    Standalone illegal data detection function.
    Args:
        df: DataFrame to check
        *args, **kwargs: Passed to IllegalDataDetector
    Returns:
        List of detected violations or issues
    """
    detector = IllegalDataDetector(*args, **kwargs)
    return detector.detect(df)
