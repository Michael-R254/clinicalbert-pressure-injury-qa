import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .config import PIPELINE_DIRS
from .schemas import ChunkMetadata, RawQAPair, FilteredQAPair
from .chunking import load_chunks

logger = logging.getLogger(__name__)


def _build_chunk_map(chunks: List[ChunkMetadata]) -> Dict[str, str]:
    return {c.chunk_id: c.text for c in chunks}


def validate_answer_spans(
    qa_pairs: List[RawQAPair],
    chunks: List[ChunkMetadata],
) -> Tuple[List[FilteredQAPair], List[FilteredQAPair], List[RawQAPair]]:
    """
    Validate answer_start indices for all answerable QA pairs.

    Returns:
        valid     — pairs that passed or were repaired
        repaired  — subset of valid that needed index correction
        rejected  — pairs whose answer_text was not found in the context
    """
    chunk_map = _build_chunk_map(chunks)
    valid: List[FilteredQAPair] = []
    repaired: List[FilteredQAPair] = []
    rejected: List[RawQAPair] = []

    for qa in qa_pairs:
        if qa.is_impossible:
            # Normalise unanswerable entries
            fqa = FilteredQAPair(
                **qa.model_dump(),
                was_rephrased=False,
                validation_status="valid",
            )
            fqa.answer_text = ""
            fqa.answer_start = -1
            valid.append(fqa)
            continue

        context = chunk_map.get(qa.chunk_id)
        if context is None:
            logger.warning(f"[REJECT] {qa.qa_id}: chunk_id not found")
            rejected.append(qa)
            continue

        # Check 1: stated position is correct
        end = qa.answer_start + len(qa.answer_text)
        if (
            0 <= qa.answer_start < len(context)
            and context[qa.answer_start:end] == qa.answer_text
        ):
            valid.append(FilteredQAPair(
                **qa.model_dump(), was_rephrased=False, validation_status="valid"
            ))
            continue

        # Check 2: answer_text exists elsewhere in context
        idx = context.find(qa.answer_text)
        if idx != -1:
            fqa = FilteredQAPair(
                **{**qa.model_dump(), "answer_start": idx},
                was_rephrased=False,
                validation_status="repaired",
            )
            valid.append(fqa)
            repaired.append(fqa)
            continue

        # Check 3: stripped version
        stripped = qa.answer_text.strip()
        if stripped:
            idx = context.find(stripped)
            if idx != -1:
                fqa = FilteredQAPair(
                    **{**qa.model_dump(), "answer_text": stripped, "answer_start": idx},
                    was_rephrased=False,
                    validation_status="repaired",
                )
                valid.append(fqa)
                repaired.append(fqa)
                continue

        # All checks failed
        logger.info(f"[REJECT] {qa.qa_id}: answer_text not found in context")
        rejected.append(qa)

    return valid, repaired, rejected


def run_validation(chunk_size: int, qa_pairs: List[RawQAPair]) -> List[FilteredQAPair]:
    """
    Run answer-span validation for a pipeline.
    Saves validated pairs to pipeline_{chunk_size}/qa_filtered/qa_validated.jsonl
    and a validation report to logs/validation_report.json.
    Returns the validated (valid + repaired) list.
    """
    chunks = load_chunks(chunk_size)
    valid, repaired, rejected = validate_answer_spans(qa_pairs, chunks)

    # Save validated pairs
    out_dir = PIPELINE_DIRS[chunk_size] / "qa_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_validated.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in valid:
            f.write(pair.model_dump_json() + "\n")

    # Save report
    report = {
        "chunk_size": chunk_size,
        "total_input": len(qa_pairs),
        "valid_first_pass": len(valid) - len(repaired),
        "repaired": len(repaired),
        "rejected": len(rejected),
        "retained": len(valid),
        "rejected_ids": [qa.qa_id for qa in rejected],
    }
    report_path = PIPELINE_DIRS[chunk_size] / "logs" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[chunk_size={chunk_size}] Validation: "
        f"{report['valid_first_pass']} valid, "
        f"{report['repaired']} repaired, "
        f"{report['rejected']} rejected → {out_path}"
    )
    return valid


def load_validated_pairs(chunk_size: int) -> List[FilteredQAPair]:
    path = PIPELINE_DIRS[chunk_size] / "qa_filtered" / "qa_validated.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No validated pairs at {path}.")
    with open(path, encoding="utf-8") as f:
        return [FilteredQAPair.model_validate_json(line) for line in f if line.strip()]


def assert_all_spans_valid(pairs: List[FilteredQAPair], chunks: List[ChunkMetadata]) -> None:
    """Hard assertion: every answerable pair must satisfy the span invariant."""
    chunk_map = _build_chunk_map(chunks)
    failures = 0
    for qa in pairs:
        if qa.is_impossible:
            continue
        context = chunk_map[qa.chunk_id]
        end = qa.answer_start + len(qa.answer_text)
        if context[qa.answer_start:end] != qa.answer_text:
            print(f"[FAIL] {qa.qa_id}: span mismatch")
            failures += 1
    if failures:
        raise AssertionError(f"{failures} answer span(s) failed the invariant check.")
    print(f"All {sum(1 for q in pairs if not q.is_impossible)} answerable spans validated.")
