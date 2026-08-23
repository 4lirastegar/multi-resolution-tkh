"""Central configuration: paths, snapshot years, level budgets, seeds, model names."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"

TKH_JSON = DATA_DIR / "tkh_collection10.json"
QUESTIONS_CSV = DATA_DIR / "questions.csv"
GROUND_TRUTH_JSON = DATA_DIR / "ground_truth.json"
ARTICLES_CSV = DATA_DIR / "collection10_articles.csv"

# Temporal snapshots: H(t) = everything with year <= t.
# Chosen so each snapshot is well-populated (edge counts ~405/589/1063/1429).
SNAPSHOT_YEARS = [2020, 2022, 2024, 2026]

# P2 size budgets: max super-nodes per level (level 0 = coarsest).
LEVEL_BUDGETS = [12, 60, 300]

# Reproducibility: the five seeds used for perturbation/stability runs (T6).
SEEDS = [0, 1, 2, 3, 4]

# Embedding used to BUILD clusters (semantic signal, P3).
CLUSTERING_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Independent embedding used only to EVALUATE coherence (T6 anti-circularity).
# Must be a different model family from the clustering embedding.
EVALUATION_EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"

# Weight of semantic signal vs structural signal in the combined affinity (P3).
# alpha=1 → pure semantic; alpha=0 → pure structural.
ALPHA = 0.6

# Jaccard overlap threshold for matching supernodes across snapshots (T3).
JACCARD_MATCH_THRESHOLD = 0.3
