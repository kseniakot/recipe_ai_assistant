QDRANT_URL = "http://localhost:6333"
COLLECTION = "recipes"

CSV_PATH = "/Users/quantik/Documents/epic/genAi/data/RAW_recipes.csv"
SAMPLE_SIZE = 500

DENSE_DIM = 1024  # BGE-M3 dense vector size

# Drop reranked results below this score
RERANK_MIN_SCORE = 0.03

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "meta-llama-3.1-8b-instruct"
