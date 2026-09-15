import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from .config import (
    SOURCES_RAW_DIR, SOURCES_METADATA, PIPELINE_DIRS, CHUNK_CONFIGS,
)
from .schemas import ChunkMetadata, SourceMetadata


def load_sources() -> Dict[str, Tuple[SourceMetadata, str]]:
    """Return {source_id: (metadata, raw_text)} for all registered sources."""
    with open(SOURCES_METADATA, encoding="utf-8") as f:
        entries = json.load(f)

    sources = {}
    for entry in entries:
        meta = SourceMetadata(**entry)
        txt_path = SOURCES_RAW_DIR / f"{meta.source_id}.txt"
        if not txt_path.exists():
            print(f"[WARN] Source file not found: {txt_path}")
            continue
        text = txt_path.read_text(encoding="utf-8")
        sources[meta.source_id] = (meta, text)

    return sources


def _clean_text(text: str) -> str:
    """Normalise whitespace and encoding artefacts."""
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _snap_to_sentence_boundary(chunk_text: str, full_text: str) -> str:
    """
    Extend or trim the chunk end to the nearest sentence-ending punctuation
    within ±15% of the chunk length, measured from the chunk end position.
    """
    end_pos = full_text.find(chunk_text)
    if end_pos == -1:
        return chunk_text

    approx_end = end_pos + len(chunk_text)
    tolerance = max(20, int(len(chunk_text) * 0.15))

    search_start = max(0, approx_end - tolerance)
    search_end = min(len(full_text), approx_end + tolerance)
    window = full_text[search_start:search_end]

    best_pos = None
    best_distance = tolerance + 1

    for m in re.finditer(r"[.?!](?:\s|$)", window):
        abs_pos = search_start + m.end()
        dist = abs(abs_pos - approx_end)
        if dist < best_distance:
            best_distance = dist
            best_pos = abs_pos

    if best_pos is not None:
        snapped = full_text[end_pos:best_pos].strip()
        if snapped:
            return snapped

    return chunk_text


def chunk_document(
    text: str,
    source_id: str,
    source_category: str,
    chunk_size: int,
    overlap: int,
) -> List[ChunkMetadata]:
    """Split a document into overlapping word-based chunks with sentence snapping."""
    text = _clean_text(text)
    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for i, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start: start + chunk_size]

        if len(chunk_words) < chunk_size * 0.4:
            break  # discard very short trailing fragments

        chunk_text = " ".join(chunk_words)
        chunk_text = _snap_to_sentence_boundary(chunk_text, text)

        chunks.append(
            ChunkMetadata(
                chunk_id=f"{source_id}_c{i:04d}",
                source_id=source_id,
                source_category=source_category,
                chunk_index=i,
                text=chunk_text,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                chunk_size_config=chunk_size,
                overlap_config=overlap,
            )
        )

    return chunks


def run_chunking_pipeline(chunk_size: int) -> List[ChunkMetadata]:
    """
    Run the full chunking pipeline for the given chunk_size.
    Saves chunks to pipeline_{chunk_size}/chunks/chunks.jsonl.
    Returns the list of ChunkMetadata objects.
    """
    cfg = CHUNK_CONFIGS[chunk_size]
    out_dir = PIPELINE_DIRS[chunk_size] / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"

    sources = load_sources()
    if not sources:
        raise RuntimeError(
            f"No source documents found in {SOURCES_RAW_DIR}. "
            "Please add .txt files before running the pipeline."
        )

    all_chunks: List[ChunkMetadata] = []
    for source_id, (meta, text) in sources.items():
        doc_chunks = chunk_document(
            text=text,
            source_id=source_id,
            source_category=meta.category,
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap"],
        )
        all_chunks.extend(doc_chunks)
        print(f"  {source_id}: {len(doc_chunks)} chunks")

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    print(f"[chunk_size={chunk_size}] {len(all_chunks)} chunks saved → {out_path}")
    return all_chunks


def load_chunks(chunk_size: int) -> List[ChunkMetadata]:
    """Load previously saved chunks from disk."""
    path = PIPELINE_DIRS[chunk_size] / "chunks" / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No chunks found at {path}. Run run_chunking_pipeline first.")
    with open(path, encoding="utf-8") as f:
        return [ChunkMetadata.model_validate_json(line) for line in f if line.strip()]
