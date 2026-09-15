import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from .config import PIPELINE_DIRS, DIVERSITY_THRESHOLD, EMBEDDING_MODEL
from .schemas import FilteredQAPair
from .qa_generation import rephrase_question


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(a.reshape(1, -1), b)[0]


def diversity_filter(
    qa_pairs: List[FilteredQAPair],
    threshold: float = DIVERSITY_THRESHOLD,
    ollama_model: Optional[str] = None,
) -> Tuple[List[FilteredQAPair], List[FilteredQAPair], int]:
    """
    Semantic diversity filter with rephrase-before-discard strategy.

    For each QA pair:
    1. Embed the question.
    2. If max cosine similarity to all kept questions < threshold → keep.
    3. Else rephrase question and re-check.
    4. If still too similar → discard.

    Returns:
        kept          — diverse QA pairs to retain
        discarded     — removed near-duplicates
        rephrased_count — number of pairs saved by rephrasing
    """
    model = _load_embedding_model()
    kept: List[FilteredQAPair] = []
    discarded: List[FilteredQAPair] = []
    kept_embeddings: List[np.ndarray] = []
    rephrased_count = 0

    for qa in tqdm(qa_pairs, desc="Diversity filtering"):
        question_text = qa.question
        emb = model.encode([question_text])[0]

        if not kept_embeddings:
            kept.append(qa)
            kept_embeddings.append(emb)
            continue

        kept_matrix = np.array(kept_embeddings)
        sims = _cosine_similarity_matrix(emb, kept_matrix)
        max_sim = float(sims.max())

        if max_sim < threshold:
            kept.append(qa)
            kept_embeddings.append(emb)
        else:
            # Try rephrase
            new_question = rephrase_question(question_text, model=ollama_model) if ollama_model else question_text
            new_emb = model.encode([new_question])[0]
            new_sims = _cosine_similarity_matrix(new_emb, kept_matrix)
            new_max_sim = float(new_sims.max())

            if new_max_sim < threshold:
                qa = qa.model_copy(update={"question": new_question, "was_rephrased": True})
                kept.append(qa)
                kept_embeddings.append(new_emb)
                rephrased_count += 1
            else:
                discarded.append(qa)

    return kept, discarded, rephrased_count


def run_diversity_filter(
    chunk_size: int,
    qa_pairs: List[FilteredQAPair],
    threshold: float = DIVERSITY_THRESHOLD,
    ollama_model: Optional[str] = None,
) -> List[FilteredQAPair]:
    """
    Run diversity filtering for a pipeline.
    Saves filtered pairs to pipeline_{chunk_size}/qa_filtered/qa_filtered.jsonl
    and a diversity report to logs/diversity_report.json.
    """
    kept, discarded, rephrased_count = diversity_filter(
        qa_pairs, threshold=threshold, ollama_model=ollama_model
    )

    out_dir = PIPELINE_DIRS[chunk_size] / "qa_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_filtered.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in kept:
            f.write(pair.model_dump_json() + "\n")

    report = {
        "chunk_size": chunk_size,
        "threshold": threshold,
        "input_pairs": len(qa_pairs),
        "kept": len(kept),
        "discarded": len(discarded),
        "rephrased_and_kept": rephrased_count,
    }
    report_path = PIPELINE_DIRS[chunk_size] / "logs" / "diversity_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[chunk_size={chunk_size}] Diversity filter: "
        f"{len(kept)} kept ({rephrased_count} rephrased), "
        f"{len(discarded)} discarded → {out_path}"
    )
    return kept


def load_filtered_pairs(chunk_size: int) -> List[FilteredQAPair]:
    path = PIPELINE_DIRS[chunk_size] / "qa_filtered" / "qa_filtered.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No filtered pairs at {path}.")
    with open(path, encoding="utf-8") as f:
        return [FilteredQAPair.model_validate_json(line) for line in f if line.strip()]
