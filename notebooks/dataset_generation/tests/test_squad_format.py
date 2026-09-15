import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pressure_ulcer_qa.schemas import ChunkMetadata, FilteredQAPair
from pressure_ulcer_qa.squad_formatter import assemble_squad_json, validate_squad_schema


CONTEXT = "A Stage III pressure injury involves full-thickness skin loss."


def _chunk():
    return ChunkMetadata(
        chunk_id="SRC_A01_c0000", source_id="SRC_A01", source_category="A",
        chunk_index=0, text=CONTEXT,
        word_count=len(CONTEXT.split()), char_count=len(CONTEXT),
        chunk_size_config=250, overlap_config=50,
    )


def _answerable_qa():
    return FilteredQAPair(
        qa_id="SRC_A01_c0000_q00", chunk_id="SRC_A01_c0000", source_id="SRC_A01",
        question="What type of skin loss does a Stage III injury involve?",
        answer_text="full-thickness skin loss",
        answer_start=CONTEXT.index("full-thickness skin loss"),
        is_impossible=False,
        question_type="DEFINITIONAL", clinical_topic="classification",
        generation_model="test", generation_timestamp="2026-01-01T00:00:00+00:00",
        was_rephrased=False, validation_status="valid",
    )


def _unanswerable_qa():
    return FilteredQAPair(
        qa_id="SRC_A01_c0000_q01", chunk_id="SRC_A01_c0000", source_id="SRC_A01",
        question="What antibiotic is recommended for Stage III injuries?",
        answer_text="", answer_start=-1, is_impossible=True,
        question_type="IMPOSSIBLE", clinical_topic="wound_management",
        generation_model="test", generation_timestamp="2026-01-01T00:00:00+00:00",
        was_rephrased=False, validation_status="valid",
    )


def test_squad_structure():
    dataset = assemble_squad_json([_answerable_qa(), _unanswerable_qa()], [_chunk()])
    assert dataset.version == "v2.0"
    assert len(dataset.data) == 1
    article = dataset.data[0]
    assert len(article.paragraphs) == 1
    para = article.paragraphs[0]
    assert para.context == CONTEXT
    assert len(para.qas) == 2


def test_answerable_qa_format():
    dataset = assemble_squad_json([_answerable_qa()], [_chunk()])
    qa = dataset.data[0].paragraphs[0].qas[0]
    assert not qa.is_impossible
    assert len(qa.answers) == 1
    assert qa.answers[0].text == "full-thickness skin loss"


def test_unanswerable_qa_format():
    dataset = assemble_squad_json([_unanswerable_qa()], [_chunk()])
    qa = dataset.data[0].paragraphs[0].qas[0]
    assert qa.is_impossible
    assert qa.answers == []


def test_validation_passes():
    dataset = assemble_squad_json([_answerable_qa(), _unanswerable_qa()], [_chunk()])
    errors = validate_squad_schema(dataset)
    assert errors == [], f"Unexpected errors: {errors}"


def test_duplicate_ids_flagged():
    qa1 = _answerable_qa()
    qa2 = _answerable_qa()  # same id
    dataset = assemble_squad_json([qa1, qa2], [_chunk()])
    errors = validate_squad_schema(dataset)
    assert any("Duplicate" in e for e in errors)
