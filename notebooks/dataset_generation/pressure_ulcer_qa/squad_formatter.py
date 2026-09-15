import json
import re
from pathlib import Path
from typing import Dict, List

from .config import PIPELINE_DIRS
from .schemas import (
    ChunkMetadata, FilteredQAPair,
    SQuADAnswer, SQuADQA, SQuADParagraph, SQuADArticle, SQuADDataset,
)


def _section_slug(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", source_id)


def assemble_squad_json(
    qa_pairs: List[FilteredQAPair],
    chunks: List[ChunkMetadata],
    version: str = "v2.0",
) -> SQuADDataset:
    """
    Assemble QA pairs and chunks into an official SQuAD v2 dataset structure.

    Groups chunks by source_id to form SQuAD "articles" (title entries).
    Only includes chunks that have at least one QA pair.
    """
    # Index QA pairs by chunk_id
    qa_by_chunk: Dict[str, List[FilteredQAPair]] = {}
    for qa in qa_pairs:
        qa_by_chunk.setdefault(qa.chunk_id, []).append(qa)

    # Group chunks by source section (title)
    chunks_by_source: Dict[str, List[ChunkMetadata]] = {}
    for chunk in chunks:
        title = _section_slug(chunk.source_id)
        chunks_by_source.setdefault(title, []).append(chunk)

    articles: List[SQuADArticle] = []

    for title, title_chunks in chunks_by_source.items():
        paragraphs: List[SQuADParagraph] = []

        for chunk in title_chunks:
            chunk_qas = qa_by_chunk.get(chunk.chunk_id, [])
            if not chunk_qas:
                continue

            qas: List[SQuADQA] = []
            for qa in chunk_qas:
                if qa.is_impossible:
                    squad_qa = SQuADQA(
                        question=qa.question,
                        id=qa.qa_id,
                        answers=[],
                        is_impossible=True,
                    )
                else:
                    squad_qa = SQuADQA(
                        question=qa.question,
                        id=qa.qa_id,
                        answers=[SQuADAnswer(text=qa.answer_text, answer_start=qa.answer_start)],
                        is_impossible=False,
                    )
                qas.append(squad_qa)

            paragraphs.append(SQuADParagraph(context=chunk.text, qas=qas))

        if paragraphs:
            articles.append(SQuADArticle(title=title, paragraphs=paragraphs))

    return SQuADDataset(version=version, data=articles)


def validate_squad_schema(dataset: SQuADDataset) -> List[str]:
    """
    Validate a SQuADDataset and return a list of error messages (empty = valid).
    Checks: unique IDs, non-empty contexts, answer_start within bounds.
    """
    errors: List[str] = []
    seen_ids = set()

    for article in dataset.data:
        for para in article.paragraphs:
            if not para.context.strip():
                errors.append(f"Empty context in article '{article.title}'")
            for qa in para.qas:
                if qa.id in seen_ids:
                    errors.append(f"Duplicate QA id: {qa.id}")
                seen_ids.add(qa.id)

                if not qa.is_impossible:
                    for ans in qa.answers:
                        end = ans.answer_start + len(ans.text)
                        if para.context[ans.answer_start:end] != ans.text:
                            errors.append(
                                f"Span mismatch in {qa.id}: "
                                f"'{ans.text}' not at index {ans.answer_start}"
                            )
                else:
                    if qa.answers:
                        errors.append(f"{qa.id} is_impossible=true but has non-empty answers")

    return errors


def save_squad_json(dataset: SQuADDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)


def run_squad_assembly(
    chunk_size: int,
    qa_pairs: List[FilteredQAPair],
    chunks: List[ChunkMetadata],
) -> SQuADDataset:
    """
    Assemble, validate, and save the full SQuAD v2 JSON for a pipeline.
    """
    dataset = assemble_squad_json(qa_pairs, chunks)

    errors = validate_squad_schema(dataset)
    if errors:
        print(f"[chunk_size={chunk_size}] Schema validation WARNINGS ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print(f"[chunk_size={chunk_size}] Schema validation passed.")

    total_qas = sum(
        len(para.qas)
        for article in dataset.data
        for para in article.paragraphs
    )

    out_path = PIPELINE_DIRS[chunk_size] / "squad" / f"squad_{chunk_size}_full.json"
    save_squad_json(dataset, out_path)
    print(f"[chunk_size={chunk_size}] {total_qas} QA pairs → {out_path}")
    return dataset


def load_squad_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
