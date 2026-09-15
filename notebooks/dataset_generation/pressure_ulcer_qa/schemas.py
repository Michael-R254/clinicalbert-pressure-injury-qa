from pydantic import BaseModel, field_validator
from typing import List, Optional


class SourceMetadata(BaseModel):
    source_id: str
    category: str               # "A", "B", or "C"
    title: str
    sections_used: str
    year: int
    access_date: str
    url_or_doi: str
    licence_notes: str


class ChunkMetadata(BaseModel):
    chunk_id: str               # e.g. "SRC_A01_c0042"
    source_id: str
    source_category: str        # "A", "B", or "C"
    chunk_index: int
    text: str                   # becomes SQuAD "context"
    word_count: int
    char_count: int
    chunk_size_config: int      # 250 / 300 / 400
    overlap_config: int         # 50 / 60 / 80


class RawQAPair(BaseModel):
    qa_id: str                  # "{chunk_id}_q{index}"
    chunk_id: str
    source_id: str
    question: str
    answer_text: str            # "" if unanswerable
    answer_start: int           # -1 if unanswerable
    is_impossible: bool
    question_type: str
    clinical_topic: str
    generation_model: str
    generation_timestamp: str

    @field_validator("answer_text")
    @classmethod
    def empty_if_impossible(cls, v: str, info) -> str:
        return v


class FilteredQAPair(RawQAPair):
    was_rephrased: bool = False
    validation_status: str = "valid"   # "valid", "repaired", "rejected"


class SQuADAnswer(BaseModel):
    text: str
    answer_start: int


class SQuADQA(BaseModel):
    question: str
    id: str
    answers: List[SQuADAnswer]
    is_impossible: bool


class SQuADParagraph(BaseModel):
    context: str
    qas: List[SQuADQA]


class SQuADArticle(BaseModel):
    title: str
    paragraphs: List[SQuADParagraph]


class SQuADDataset(BaseModel):
    version: str = "v2.0"
    data: List[SQuADArticle]
