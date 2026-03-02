"""Tests for EmbeddingConfig - pluggable embeddings for vector backends."""

from __future__ import annotations

import os
import tempfile
import uuid

import pytest

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import ChromaConfig, EmbeddingConfig, Memory, QdrantConfig


def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding for tests. Returns 384 dimensions."""
    vals = [float(ord(c) % 10) / 10.0 for c in (text + "\0" * 16)[:16]]
    return vals + [0.0] * (384 - len(vals))


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_defaults(self) -> None:
        cfg = EmbeddingConfig()
        assert cfg.model == "text-embedding-3-small"
        assert cfg.provider == "openai"
        assert cfg.dimensions == 1536

    def test_custom_fn(self) -> None:
        cfg = EmbeddingConfig(custom_fn=_fake_embedding, dimensions=384)
        result = cfg.embed("hello")
        assert len(result) == 384
        assert result[0] == 0.4  # 'h' -> ord 104 % 10 / 10 = 0.4

    def test_custom_fn_dimensions_mismatch_raises(self) -> None:
        def wrong_size(text: str) -> list[float]:
            return [1.0, 2.0]  # Too short

        cfg = EmbeddingConfig(custom_fn=wrong_size, dimensions=384)
        with pytest.raises(ValueError, match="returned 2 dimensions"):
            cfg.embed("test")


class TestMemoryWithEmbeddingConfig:
    """Memory with EmbeddingConfig in Qdrant/Chroma."""

    @pytest.fixture
    def temp_dir(self) -> str:
        d = tempfile.mkdtemp()
        yield d
        import shutil

        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    def test_qdrant_with_custom_embedding(self, temp_dir: str) -> None:
        """Qdrant with EmbeddingConfig.custom_fn uses custom embeddings."""
        mem = Memory(
            backend=MemoryBackend.QDRANT,
            write_mode=WriteMode.SYNC,
            qdrant=QdrantConfig(
                path=temp_dir,
                collection="test_emb",
                embedding_config=EmbeddingConfig(
                    custom_fn=_fake_embedding,
                    dimensions=384,
                ),
            ),
        )
        mem.remember("Custom embedding test", memory_type=MemoryType.CORE)
        results = mem.recall(query="Custom", count=5)
        assert len(results) >= 1

    def test_chroma_with_custom_embedding(self, temp_dir: str) -> None:
        """Chroma with EmbeddingConfig.custom_fn uses custom embeddings."""
        chroma_path = os.path.join(temp_dir, f"chroma_{uuid.uuid4().hex[:8]}")
        mem = Memory(
            backend=MemoryBackend.CHROMA,
            write_mode=WriteMode.SYNC,
            chroma=ChromaConfig(
                path=chroma_path,
                collection="test_emb",
                embedding_config=EmbeddingConfig(
                    custom_fn=_fake_embedding,
                    dimensions=384,
                ),
            ),
        )
        mem.remember("Chroma custom embedding", memory_type=MemoryType.EPISODIC)
        results = mem.recall(query="Chroma", count=5)
        assert len(results) >= 1
