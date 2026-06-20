from openai import OpenAI

from config import LLM_MODEL

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
