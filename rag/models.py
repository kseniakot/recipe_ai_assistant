from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel


@lru_cache(maxsize=1)
def get_embedding_model() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
