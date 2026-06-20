from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
from rag.reranker import CrossEncoderReranker

from rag.config import LLM_BASE_URL


@lru_cache(maxsize=1)
def get_large_language_model() -> OpenAI:
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    return client


@lru_cache(maxsize=1)
def get_embedding_model() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
