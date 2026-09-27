"""The access decision, with CLIP replaced by fixed vectors and an in-memory database."""

import uuid

import chromadb
import numpy as np
import pytest

import verification
from config import FaceAuthConfig
from verification import COSINE_CONFIGURATION, verify_identity

ANY_CROP = np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def config() -> FaceAuthConfig:
    return FaceAuthConfig(_env_file=None, verification_threshold=0.8)


@pytest.fixture
def collection():
    # A unique name per test: in-memory clients can share state inside one process
    collection = chromadb.EphemeralClient().create_collection(
        name=f"test_{uuid.uuid4().hex}", configuration=COSINE_CONFIGURATION
    )
    collection.add(ids=["alice_001"], embeddings=[[1.0, 0.0]], metadatas=[{"name": "Alice"}])
    return collection


def use_query_embedding(monkeypatch, vector: list[float]) -> None:
    monkeypatch.setattr(verification, "embed_face", lambda crop, config: vector)


def test_close_embedding_is_granted(monkeypatch, collection, config):
    use_query_embedding(monkeypatch, [0.99, 0.14])  # about 8 degrees away: similarity ≈ 0.99

    result = verify_identity(ANY_CROP, collection, config)

    assert result.verified
    assert result.matched_name == "Alice"
    assert result.similarity == pytest.approx(0.99, abs=0.01)


def test_distant_embedding_is_denied_but_reports_closest_user(monkeypatch, collection, config):
    use_query_embedding(monkeypatch, [0.6, 0.8])  # similarity 0.6, below the 0.8 threshold

    result = verify_identity(ANY_CROP, collection, config)

    assert not result.verified
    assert result.matched_name == "Alice"
    assert result.similarity == pytest.approx(0.6, abs=0.01)


def test_empty_database_is_denied(monkeypatch, config):
    empty = chromadb.EphemeralClient().create_collection(
        name=f"test_{uuid.uuid4().hex}", configuration=COSINE_CONFIGURATION
    )
    use_query_embedding(monkeypatch, [1.0, 0.0])

    result = verify_identity(ANY_CROP, empty, config)

    assert not result.verified
    assert "enroll.py" in result.reason
