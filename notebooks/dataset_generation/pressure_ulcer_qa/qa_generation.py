import json
import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import (
    PIPELINE_DIRS, OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL, OLLAMA_TEMPERATURE,
    QA_PAIRS_PER_LARGE_CHUNK, QA_PAIRS_PER_SMALL_CHUNK,
    ANSWERABLE_LARGE, UNANSWERABLE_LARGE,
    ANSWERABLE_SMALL, UNANSWERABLE_SMALL,
    CLINICAL_TOPICS, QUESTION_TYPES,
)
from .schemas import ChunkMetadata, RawQAPair
from .chunking import load_chunks

logger = logging.getLogger(__name__)

_QUESTION_TYPES_STR = ", ".join(QUESTION_TYPES)
_CLINICAL_TOPICS_STR = ", ".join(CLINICAL_TOPICS)

PROMPT_TEMPLATE = """\
You are a clinical QA pair generator for a medical NLP dataset.

TASK: Given the clinical text chunk below, generate exactly {{n_pairs}} question-answer pairs.

STRICT RULES:
1. Generate {{n_answerable}} ANSWERABLE and {{n_unanswerable}} UNANSWERABLE questions.

ANSWERABLE rules:
- answer_text MUST be copied character-for-character as an exact substring from the chunk.
- answer_start MUST be the integer character index of the first character of answer_text in the chunk.
- Verify: chunk[answer_start : answer_start + len(answer_text)] == answer_text
- Vary question types: FACTUAL, DEFINITIONAL, PROCEDURAL, CAUSAL, COMPARATIVE, NUMERICAL.

UNANSWERABLE rules:
- Question must be clinically plausible and topically related to the chunk.
- The answer CANNOT be found in or inferred from the chunk.
- Set is_impossible to true, answers to empty list [].
- Do NOT use nonsensical or completely off-topic questions.

General rules:
- Each question must be self-contained (understandable without reading the chunk).
- Do NOT start every question with "What".
- question_type must be one of: {question_types}
- clinical_topic must be one of: {clinical_topics}

CLINICAL TEXT CHUNK:
---
{{chunk_text}}
---

Respond with ONLY a valid JSON array. No text before or after. Each element:
[
  {{{{
    "question": "...",
    "answer_text": "...",
    "answer_start": <int or -1>,
    "is_impossible": false,
    "question_type": "...",
    "clinical_topic": "..."
  }}}}
]
""".format(question_types=_QUESTION_TYPES_STR, clinical_topics=_CLINICAL_TOPICS_STR)

REPHRASE_PROMPT = """\
Rephrase the following clinical question using substantially different wording \
and sentence structure while preserving the exact clinical meaning. \
Return ONLY the rephrased question, nothing else.

Original: {question}
"""


def _build_prompt(chunk_text: str) -> Tuple[str, int, int]:
    word_count = len(chunk_text.split())
    if word_count >= 150:
        n_pairs = QA_PAIRS_PER_LARGE_CHUNK
        n_answerable = ANSWERABLE_LARGE
        n_unanswerable = UNANSWERABLE_LARGE
    else:
        n_pairs = QA_PAIRS_PER_SMALL_CHUNK
        n_answerable = ANSWERABLE_SMALL
        n_unanswerable = UNANSWERABLE_SMALL

    prompt = PROMPT_TEMPLATE.format(
        n_pairs=n_pairs,
        n_answerable=n_answerable,
        n_unanswerable=n_unanswerable,
        chunk_text=chunk_text,
    )
    return prompt, n_answerable, n_unanswerable


def _call_ollama(prompt: str, model: str, temperature: float = OLLAMA_TEMPERATURE) -> str:
    """Call Ollama and return the response string."""
    try:
        import ollama
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response["response"]
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}") from e


def _extract_json_array(raw: str) -> List[dict]:
    """Extract a JSON array from a model response, even with surrounding text."""
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON array found in response:\n{raw[:300]}")


def _validate_raw_item(item: dict) -> bool:
    required = {"question", "answer_text", "answer_start", "is_impossible",
                 "question_type", "clinical_topic"}
    return required.issubset(item.keys())


def generate_qa_pairs(
    chunk: ChunkMetadata,
    model: str = OLLAMA_MODEL,
    max_retries: int = 3,
) -> List[RawQAPair]:
    """Generate QA pairs for a single chunk. Returns empty list on failure."""
    prompt, n_answerable, n_unanswerable = _build_prompt(chunk.text)
    timestamp = datetime.now(timezone.utc).isoformat()

    for attempt in range(max_retries):
        try:
            raw_response = _call_ollama(prompt, model)
            items = _extract_json_array(raw_response)
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                logger.warning(f"[SKIP] {chunk.chunk_id}: failed after {max_retries} attempts — {e}")
                return []
            # On retry, add explicit JSON instruction
            prompt = "Output ONLY valid JSON, no commentary.\n\n" + prompt
            time.sleep(1)
            continue

        pairs = []
        for idx, item in enumerate(items):
            if not _validate_raw_item(item):
                continue
            try:
                pair = RawQAPair(
                    qa_id=f"{chunk.chunk_id}_q{idx:02d}",
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    question=str(item["question"]).strip(),
                    answer_text=str(item.get("answer_text", "")).strip(),
                    answer_start=int(item.get("answer_start", -1)),
                    is_impossible=bool(item["is_impossible"]),
                    question_type=str(item.get("question_type", "FACTUAL")).upper(),
                    clinical_topic=str(item.get("clinical_topic", "other")).lower(),
                    generation_model=model,
                    generation_timestamp=timestamp,
                )
                pairs.append(pair)
            except Exception as e:
                logger.debug(f"Skipping malformed item in {chunk.chunk_id}: {e}")

        return pairs

    return []


def run_qa_generation(chunk_size: int, model: str = OLLAMA_MODEL) -> List[RawQAPair]:
    """
    Generate QA pairs for every chunk in the given pipeline.
    Saves results to pipeline_{chunk_size}/qa_raw/qa_raw.jsonl.
    Returns all generated RawQAPair objects.
    """
    out_dir = PIPELINE_DIRS[chunk_size] / "qa_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_raw.jsonl"

    log_path = PIPELINE_DIRS[chunk_size] / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    chunks = load_chunks(chunk_size)
    all_pairs: List[RawQAPair] = []
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc=f"Generating QA (chunk_size={chunk_size})"):
            pairs = generate_qa_pairs(chunk, model=model)
            if not pairs:
                skipped += 1
            for pair in pairs:
                f.write(pair.model_dump_json() + "\n")
            all_pairs.extend(pairs)

    print(
        f"[chunk_size={chunk_size}] {len(all_pairs)} raw QA pairs generated "
        f"({skipped} chunks skipped) → {out_path}"
    )
    return all_pairs


def load_raw_qa_pairs(chunk_size: int) -> List[RawQAPair]:
    """Load previously saved raw QA pairs from disk."""
    path = PIPELINE_DIRS[chunk_size] / "qa_raw" / "qa_raw.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No raw QA pairs at {path}.")
    with open(path, encoding="utf-8") as f:
        return [RawQAPair.model_validate_json(line) for line in f if line.strip()]


def rephrase_question(question: str, model: str = OLLAMA_MODEL) -> str:
    """Ask Ollama to rephrase a question with different wording."""
    prompt = REPHRASE_PROMPT.format(question=question)
    try:
        response = _call_ollama(prompt, model, temperature=0.7)
        rephrased = response.strip().strip('"').strip("'")
        return rephrased if rephrased else question
    except Exception:
        return question
