import logging
from collections import defaultdict

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)
from FlagEmbedding import BGEM3FlagModel

from rag.config import LLM_MODEL, COLLECTION, RERANK_MIN_SCORE

logger = logging.getLogger(__name__)

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


def multi_query_dense_sparse(
    client: QdrantClient, model: BGEM3FlagModel, queries: list[str], limit: int = 20
):
    """Run a separate dense (cosine) and sparse (dot) search for every query,
    from a single encode each. Returns [(query, dense_points, sparse_points), ...].
    """
    out = []
    for q in queries:
        dense, sparse = embed_query(model, q)
        d = client.query_points(
            COLLECTION, query=dense, using="dense", limit=limit, with_payload=True
        ).points
        s = client.query_points(
            COLLECTION, query=sparse, using="sparse", limit=limit, with_payload=True
        ).points
        out.append((q, d, s))
    return out


def lookup_by_name(
    client: QdrantClient, model: BGEM3FlagModel, reranker, name: str, limit: int = 1
):
    """Find a recipe by name: exact match on the name field, else hybrid
    retrieve + cross-encoder rerank."""
    exact, _ = client.scroll(
        COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="name", match=MatchValue(value=name.strip().lower()))
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    if exact:
        logger.info("lookup_by_name %r -> exact match", name)
        return exact

    # fallback in case model used approximate name
    candidates = hybrid_search(client, model, name, limit=10)
    ranked = rerank(reranker, name, candidates, top_k=limit)
    logger.info(
        "lookup_by_name %r -> no exact match; rerank %s",
        name,
        [(p.payload["name"], round(p.score, 4)) for p in ranked],
    )
    return ranked


def reciprocal_rank_fusion(result_lists: list[list], k: int = 60, limit: int = 10):
    scores: dict = defaultdict(float)
    point_by_id: dict = {}
    for results in result_lists:
        for rank, point in enumerate(results):
            scores[point.id] += 1 / (k + rank + 1)
            point_by_id[point.id] = point
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused = []
    for pid, score in ranked[:limit]:
        point = point_by_id[pid]
        point.score = score  # overwrite inner score with outer RRF score
        fused.append(point)
    return fused


def _rerank_text(payload: dict) -> str:
    return (
        f"{payload['name']}. "
        f"Ingredients: {', '.join(payload['ingredients'])}. "
        f"Tags: {', '.join(payload['tags'])}."
    )


def rerank(
    reranker,
    query: str,
    points: list,
    top_k: int = 5,
    min_score: float = RERANK_MIN_SCORE,
):
    """Re-score with the cross-encoder, drop results below `min_score`, keep top_k."""
    if not points:
        return []
    pairs = [[query, _rerank_text(p.payload)] for p in points]
    scores = reranker.compute_score(pairs, normalize=True)
    ranked = sorted(zip(points, scores), key=lambda ps: ps[1], reverse=True)
    out = []
    for point, score in ranked:
        if score < min_score:
            break
        point.score = float(score)
        out.append(point)
        if len(out) >= top_k:
            break
    return out
