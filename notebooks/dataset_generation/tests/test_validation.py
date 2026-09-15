import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pressure_ulcer_qa.schemas import ChunkMetadata, RawQAPair
from pressure_ulcer_qa.validation import validate_answer_spans


def _make_chunk(text: str) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id="TEST_c0000",
        source_id="TEST",
        source_category="A",
        chunk_index=0,
        text=text,
        word_count=len(text.split()),
        char_count=len(text),
        chunk_size_config=250,
        overlap_config=50,
    )


def _make_qa(answer_text: str, answer_start: int, is_impossible: bool = False) -> RawQAPair:
    return RawQAPair(
        qa_id="TEST_c0000_q00",
        chunk_id="TEST_c0000",
        source_id="TEST",
        question="Test question?",
        answer_text=answer_text,
        answer_start=answer_start,
        is_impossible=is_impossible,
        question_type="FACTUAL",
        clinical_topic="classification",
        generation_model="test",
        generation_timestamp="2026-01-01T00:00:00+00:00",
    )


CONTEXT = "A Stage III pressure injury involves full-thickness skin loss."


def test_valid_pair_accepted():
    chunk = _make_chunk(CONTEXT)
    qa = _make_qa("Stage III", 2)  # correct index
    valid, repaired, rejected = validate_answer_spans([qa], [chunk])
    assert len(valid) == 1
    assert len(repaired) == 0
    assert len(rejected) == 0


def test_wrong_index_repaired():
    chunk = _make_chunk(CONTEXT)
    qa = _make_qa("Stage III", 99)  # wrong index
    valid, repaired, rejected = validate_answer_spans([qa], [chunk])
    assert len(valid) == 1
    assert len(repaired) == 1
    assert valid[0].answer_start == CONTEXT.find("Stage III")


def test_missing_answer_rejected():
    chunk = _make_chunk(CONTEXT)
    qa = _make_qa("this text is not in the context at all", 0)
    valid, repaired, rejected = validate_answer_spans([qa], [chunk])
    assert len(rejected) == 1
    assert len(valid) == 0


def test_unanswerable_normalised():
    chunk = _make_chunk(CONTEXT)
    qa = _make_qa("", -1, is_impossible=True)
    valid, repaired, rejected = validate_answer_spans([qa], [chunk])
    assert len(valid) == 1
    assert valid[0].answer_text == ""
    assert valid[0].answer_start == -1
