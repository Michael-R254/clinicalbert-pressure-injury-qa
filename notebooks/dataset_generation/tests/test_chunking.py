import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pressure_ulcer_qa.chunking import chunk_document, _clean_text


def test_chunk_count():
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_document(text, "TEST", "A", chunk_size=250, overlap=50)
    assert len(chunks) >= 2


def test_chunk_ids_unique():
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_document(text, "TEST", "A", chunk_size=250, overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_short_trailing_fragment_discarded():
    text = " ".join([f"word{i}" for i in range(260)])
    chunks = chunk_document(text, "TEST", "A", chunk_size=250, overlap=50)
    for c in chunks:
        assert c.word_count >= 100


def test_metadata_fields():
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = chunk_document(text, "SRC_A01", "A", chunk_size=250, overlap=50)
    c = chunks[0]
    assert c.source_id == "SRC_A01"
    assert c.source_category == "A"
    assert c.chunk_size_config == 250
    assert c.overlap_config == 50
    assert c.char_count == len(c.text)


def test_clean_text_normalises_whitespace():
    raw = "hello   world\r\n\r\nfoo"
    cleaned = _clean_text(raw)
    assert "   " not in cleaned
    assert "\r\n" not in cleaned
