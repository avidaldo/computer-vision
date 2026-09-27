"""
Turns a person crop into a CLIP embedding and looks for the closest enrolled identity in ChromaDB.

Sourced from face_recognition_pipeline.ipynb: embed_face() (step 3) and verify_identity() (step 5).
"""

from dataclasses import dataclass
from functools import cache
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import FaceAuthConfig

COSINE_SPACE = {"hnsw:space": "cosine"}


@dataclass
class VerificationResult:
    verified: bool
    reason: str
    matched_name: str | None = None
    similarity: float | None = None


@cache
def load_clip(model_name: str) -> SentenceTransformer:
    """Loaded once per model name and reused: loading CLIP takes seconds."""
    return SentenceTransformer(model_name)


def embed_face(bgr_crop: np.ndarray, config: FaceAuthConfig) -> list[float]:
    """OpenCV images are BGR; CLIP expects RGB. The vector is scaled to length 1 (L2 normalisation)."""
    rgb_image = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
    return load_clip(config.clip_model_name).encode(rgb_image, normalize_embeddings=True).tolist()


def verify_identity(bgr_crop: np.ndarray, collection: Any, config: FaceAuthConfig) -> VerificationResult:
    if collection.count() == 0:
        return VerificationResult(False, "no enrolled users: run enroll.py first")

    results = collection.query(
        query_embeddings=[embed_face(bgr_crop, config)],
        n_results=1,
        include=["metadatas", "distances"],
    )
    # Chroma returns cosine *distance* (0 = same direction, 2 = opposite); similarity = 1 - distance.
    similarity = 1.0 - results["distances"][0][0]
    matched_name = results["metadatas"][0][0]["name"]

    if similarity >= config.verification_threshold:
        return VerificationResult(True, "match above threshold", matched_name, similarity)
    return VerificationResult(False, f"closest match below threshold {config.verification_threshold}", matched_name, similarity)
