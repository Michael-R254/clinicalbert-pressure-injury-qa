import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .config import PIPELINE_DIRS, RANDOM_SEED
from .schemas import ChunkMetadata, FilteredQAPair, SQuADDataset
from .squad_formatter import assemble_squad_json, validate_squad_schema, save_squad_json


def split_dataset(
    chunks: List[ChunkMetadata],
    qa_pairs: List[FilteredQAPair],
    seed: int = RANDOM_SEED,
) -> Tuple[List[FilteredQAPair], List[FilteredQAPair], List[FilteredQAPair]]:
    """
    Stratified chunk-level 80/10/10 train/val/test split.

    Splits by chunk (not QA pair) to prevent data leakage from overlapping chunks.
    Stratifies by source_category (A, B, C) to ensure proportional coverage.

    Returns: (train_qa, val_qa, test_qa)
    """
    random.seed(seed)

    by_category: Dict[str, List[ChunkMetadata]] = defaultdict(list)
    for chunk in chunks:
        by_category[chunk.source_category].append(chunk)

    train_ids: Set[str] = set()
    val_ids: Set[str] = set()
    test_ids: Set[str] = set()

    for cat, cat_chunks in by_category.items():
        shuffled = cat_chunks.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.10))

        test_ids.update(c.chunk_id for c in shuffled[:n_test])
        val_ids.update(c.chunk_id for c in shuffled[n_test: n_test + n_val])
        train_ids.update(c.chunk_id for c in shuffled[n_test + n_val:])

    # Verify no chunk appears in more than one split
    assert not (train_ids & val_ids), "Leakage: train ∩ val is not empty"
    assert not (train_ids & test_ids), "Leakage: train ∩ test is not empty"
    assert not (val_ids & test_ids), "Leakage: val ∩ test is not empty"

    train_qa = [qa for qa in qa_pairs if qa.chunk_id in train_ids]
    val_qa = [qa for qa in qa_pairs if qa.chunk_id in val_ids]
    test_qa = [qa for qa in qa_pairs if qa.chunk_id in test_ids]

    return train_qa, val_qa, test_qa


def _split_summary(name: str, pairs: List[FilteredQAPair]) -> dict:
    answerable = sum(1 for q in pairs if not q.is_impossible)
    return {
        "split": name,
        "total": len(pairs),
        "answerable": answerable,
        "unanswerable": len(pairs) - answerable,
        "answerable_pct": round(answerable / len(pairs) * 100, 1) if pairs else 0,
    }


def run_split(
    chunk_size: int,
    qa_pairs: List[FilteredQAPair],
    chunks: List[ChunkMetadata],
) -> Tuple[List[FilteredQAPair], List[FilteredQAPair], List[FilteredQAPair]]:
    """
    Run train/val/test split for a pipeline.
    Saves three SQuAD v2 JSON files and a split summary to logs.
    """
    train_qa, val_qa, test_qa = split_dataset(chunks, qa_pairs)

    squad_dir = PIPELINE_DIRS[chunk_size] / "squad"
    squad_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_pairs in [("train", train_qa), ("val", val_qa), ("test", test_qa)]:
        dataset = assemble_squad_json(split_pairs, chunks)
        path = squad_dir / f"squad_{chunk_size}_{split_name}.json"
        save_squad_json(dataset, path)

        errors = validate_squad_schema(dataset)
        status = "OK" if not errors else f"{len(errors)} warnings"
        total = sum(len(p.qas) for a in dataset.data for p in a.paragraphs)
        print(f"  [{split_name}] {total} pairs, validation: {status} → {path.name}")

    # Save split summary
    summary = [_split_summary(n, p) for n, p in [("train", train_qa), ("val", val_qa), ("test", test_qa)]]
    log_path = PIPELINE_DIRS[chunk_size] / "logs" / "split_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[chunk_size={chunk_size}] Split: "
        f"train={len(train_qa)}, val={len(val_qa)}, test={len(test_qa)}"
    )
    return train_qa, val_qa, test_qa
