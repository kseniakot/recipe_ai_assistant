from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from FlagEmbedding import BGEM3FlagModel

from config import LLM_MODEL, COLLECTION

# Few-shot prompt
MULTI_QUERY_SYSTEM = (
    "You rewrite a user's recipe search query into alternative phrasings that "
    "keep the same intent but use different words, to improve recipe retrieval. "
    "Output ONLY the rephrasings, one per line, no numbering, no extra text."
)

MULTI_QUERY_EXAMPLES = [
    {
        "role": "user",
        "content": "Query: easy chicken dinner\nGive 3 rephrasings:",
    },
    {
        "role": "assistant",
        "content": (
            "quick simple chicken main dish\n"
            "fast weeknight chicken recipe\n"
            "effortless chicken meal for dinner"
        ),
    },
    {
        "role": "user",
        "content": "Query: vegan dessert without sugar\nGive 3 rephrasings:",
    },
    {
        "role": "assistant",
        "content": (
            "plant-based sugar-free sweet treat\n"
            "vegan dessert with no added sugar\n"
            "dairy-free eggless low-sugar dessert"
        ),
    },
]


def multi_query(client: OpenAI, query: str, n: int = 3) -> list[str]:
    """Expand a query into the original plus n LLM-generated rephrasings."""
    messages = (
        [{"role": "system", "content": MULTI_QUERY_SYSTEM}]
        + MULTI_QUERY_EXAMPLES
        + [{"role": "user", "content": f"Query: {query}\nGive {n} rephrasings:"}]
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.4,
    )
    raw = resp.choices[0].message.content
    variants = [line.strip() for line in raw.splitlines()]
    variants = [v for v in variants if v]
    return [query] + variants[:n]


def embed_query(model: BGEM3FlagModel, query: str):
    """Encode a query into Qdrant-ready dense vector and sparse vector."""
    out = model.encode([query], return_dense=True, return_sparse=True)
    dense = out["dense_vecs"][0].tolist()
    weights = out["lexical_weights"][0]
    sparse = SparseVector(
        indices=[int(k) for k in weights.keys()],
        values=[float(v) for v in weights.values()],
    )
    return dense, sparse


def hybrid_search(
    client: QdrantClient,
    model: BGEM3FlagModel,
    query: str,
    limit: int = 10,
    prefetch_limit: int = 20,
):
    """Dense + sparse search fused with RRF inside Qdrant."""
    dense, sparse = embed_query(model, query)
    result = client.query_points(
        COLLECTION,
        prefetch=[
            Prefetch(query=dense, using="dense", limit=prefetch_limit),
            Prefetch(query=sparse, using="sparse", limit=prefetch_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return result.points
