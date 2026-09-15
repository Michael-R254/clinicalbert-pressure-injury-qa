from pathlib import Path

ROOT = Path(__file__).parent.parent

SOURCES_DIR = ROOT / "sources"
SOURCES_RAW_DIR = SOURCES_DIR / "raw"
SOURCES_METADATA = SOURCES_DIR / "metadata.json"

PIPELINE_DIRS = {
    250: ROOT / "pipeline_250",
    300: ROOT / "pipeline_300",
    400: ROOT / "pipeline_400",
}

CHUNK_CONFIGS = {
    250: {"chunk_size": 250, "overlap": 50},
    300: {"chunk_size": 300, "overlap": 60},
    400: {"chunk_size": 400, "overlap": 80},
}

OLLAMA_MODEL = "gemma3:12b"
OLLAMA_FALLBACK_MODEL = "gemma3:4b"
OLLAMA_TEMPERATURE = 0.3

QA_PAIRS_PER_LARGE_CHUNK = 20   # chunks >= 150 words: 12 answerable + 8 unanswerable
QA_PAIRS_PER_SMALL_CHUNK = 12   # chunks < 150 words: 8 answerable + 4 unanswerable
ANSWERABLE_LARGE = 12
UNANSWERABLE_LARGE = 8
ANSWERABLE_SMALL = 8
UNANSWERABLE_SMALL = 4

DIVERSITY_THRESHOLD = 0.85
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

QUESTION_TYPES = [
    "FACTUAL", "DEFINITIONAL", "PROCEDURAL",
    "CAUSAL", "COMPARATIVE", "NUMERICAL", "IMPOSSIBLE",
]

CLINICAL_TOPICS = [
    "classification", "risk_assessment", "prevention", "nutrition",
    "repositioning", "support_surfaces", "skin_assessment",
    "wound_management", "patient_education", "documentation",
    "quality_improvement", "implementation", "pain_management", "other",
]
