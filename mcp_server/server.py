import asyncio
from typing import Literal, get_args
from dataclasses import dataclass
from contextlib import asynccontextmanager

from mcp.server.fastmcp import Context, FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from rag.config import QDRANT_URL, COLLECTION
from rag.retrieval import (
    multi_query,
    hybrid_search,
    reciprocal_rank_fusion,
    rerank,
    lookup_by_name,
)
from rag.models import get_embedding_model, get_large_language_model, get_reranker


Diet = Literal[
    "vegetarian",
    "vegan",
    "gluten-free",
    "low-carb",
    "very-low-carbs",
    "low-fat",
    "low-sodium",
    "low-cholesterol",
    "low-calorie",
    "low-saturated-fat",
    "low-protein",
    "high-protein",
    "healthy",
    "diabetic",
]
DIETS = frozenset(get_args(Diet))


def _summary(payload: dict) -> dict:
    """Compact, JSON-serializable recipe view for tool output."""
    return {
        "name": payload["name"],
        "minutes": payload["minutes"],
        "calories": payload["calories"],
        "n_ingredients": payload["n_ingredients"],
        "ingredients": payload["ingredients"],
        "tags": payload["tags"],
    }


@dataclass
class AppContext:
    client: QdrantClient
    model: object
    llm: object
    reranker: object


@asynccontextmanager
async def lifespan(server: FastMCP):
    client = QdrantClient(url=QDRANT_URL)
    ctx = AppContext(
        client=client,
        model=get_embedding_model(),
        llm=get_large_language_model(),
        reranker=get_reranker(),
    )
    try:
        yield ctx
    finally:
        client.close()


mcp = FastMCP("recipe-assistant", lifespan=lifespan)


@mcp.tool()
async def search_recipes(query: str, ctx: Context, top_k: int = 5) -> list[dict] | dict:
    """Search recipes by meaning (dish type, ingredients, constraints).

    Use for open-ended requests like "easy vegetarian dinner with beans".
    Runs multi-query expansion + hybrid (dense+sparse) search + cross-encoder
    reranking. Returns up to `top_k` recipes with name, minutes, calories,
    ingredients and tags.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    app = ctx.request_context.lifespan_context
    await ctx.info(f"Searching recipes for: {query!r}")
    try:
        # 1/3 — expand the query with the LLM
        queries = await asyncio.to_thread(multi_query, app.llm, query)
        await ctx.report_progress(1, 3, f"expanded into {len(queries)} queries")

        # 2/3 — hybrid search each variant, RRF
        result_lists = await asyncio.to_thread(
            lambda: [hybrid_search(app.client, app.model, q, limit=20) for q in queries]
        )
        fused = reciprocal_rank_fusion(result_lists, limit=20)
        await ctx.report_progress(2, 3, f"retrieved {len(fused)} candidates")

        # 3/3 — cross-encoder rerank against the original query
        points = await asyncio.to_thread(rerank, app.reranker, query, fused, top_k)
        await ctx.report_progress(3, 3, "reranked")
    except Exception as e:
        await ctx.error(f"search failed: {e}")
        return {"error": "recipe search is temporarily unavailable"}
    await ctx.info(f"Found {len(points)} recipes")
    return [_summary(p.payload) for p in points]


@mcp.tool()
async def filter_recipes(
    ctx: Context,
    diet: Diet | None = None,
    max_minutes: int | None = None,
    max_calories: float | None = None,
    limit: int = 10,
) -> list[dict] | dict:
    """Filter recipes by exact metadata (no semantic search).

    Use for hard constraints: a `diet` (one of the allowed values, e.g.
    "vegan", "gluten-free", "low-carb"), a maximum cook time in minutes, or a
    calorie ceiling. At least one filter is required. For cuisine or dish-type
    requests (e.g. "mexican breakfast") use `search_recipes` instead.
    """
    conditions = []
    if diet:
        conditions.append(FieldCondition(key="tags", match=MatchValue(value=diet)))
    if max_minutes is not None:
        conditions.append(FieldCondition(key="minutes", range=Range(lte=max_minutes)))
    if max_calories is not None:
        conditions.append(FieldCondition(key="calories", range=Range(lte=max_calories)))
    if not conditions:
        raise ValueError("provide at least one of: diet, max_minutes, max_calories")

    app = ctx.request_context.lifespan_context
    await ctx.info(
        f"Filtering recipes: diet={diet}, max_minutes={max_minutes}, max_calories={max_calories}"
    )
    try:
        points, _ = await asyncio.to_thread(
            app.client.scroll,
            COLLECTION,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )
    except Exception as e:
        await ctx.error(f"filter failed: {e}")
        return {"error": "recipe filtering is temporarily unavailable"}
    await ctx.info(f"Matched {len(points)} recipes")
    return [_summary(p.payload) for p in points]


@mcp.tool()
async def calculate_nutrition(recipe_name: str, ctx: Context) -> dict:
    """Return the nutrition breakdown for a recipe by name.

    Looks up the best-matching recipe and returns calories plus fat, sugar,
    sodium, protein, saturated fat and carbohydrates (% of daily value).
    """
    if not recipe_name or not recipe_name.strip():
        raise ValueError("recipe_name must not be empty")
    app = ctx.request_context.lifespan_context
    await ctx.info(f"Looking up nutrition for: {recipe_name!r}")
    try:
        points = await asyncio.to_thread(
            lookup_by_name, app.client, app.model, recipe_name, 1
        )
    except Exception as e:
        await ctx.error(f"nutrition lookup failed: {e}")
        return {"error": "nutrition lookup is temporarily unavailable"}
    if not points:
        await ctx.warning(f"no recipe matched {recipe_name!r}")
        return {"error": f"no recipe found for '{recipe_name}'"}
    p = points[0].payload
    return {"name": p["name"], "nutrition": p["nutrition"]}


@mcp.tool()
async def get_recipe_steps(recipe_name: str, ctx: Context) -> dict:
    """Return the cooking instructions for a recipe by name.

    Use after `search_recipes`/`filter_recipes` when the user picks a recipe and
    wants to know how to make it. Returns the ingredients and the ordered steps.
    """
    if not recipe_name or not recipe_name.strip():
        raise ValueError("recipe_name must not be empty")
    app = ctx.request_context.lifespan_context
    await ctx.info(f"Fetching steps for: {recipe_name!r}")
    try:
        points = await asyncio.to_thread(
            lookup_by_name, app.client, app.model, recipe_name, 1
        )
    except Exception as e:
        await ctx.error(f"steps lookup failed: {e}")
        return {"error": "recipe lookup is temporarily unavailable"}
    if not points:
        await ctx.warning(f"no recipe matched {recipe_name!r}")
        return {"error": f"no recipe found for '{recipe_name}'"}
    p = points[0].payload
    return {
        "name": p["name"],
        "minutes": p["minutes"],
        "ingredients": p["ingredients"],
        "steps": p["steps"],
    }


@mcp.resource("recipes://filters")
def available_filters() -> dict:
    """Filter options for `filter_recipes`: the diets that have recipes in the
    corpus, plus the available ranges for cook time (minutes) and calories."""
    client = QdrantClient(url=QDRANT_URL)
    try:
        diets = set()
        minutes, calories = [], []
        points, _ = client.scroll(COLLECTION, limit=10_000, with_payload=True)
        for p in points:
            pl = p.payload
            diets |= DIETS.intersection(pl["tags"])
            minutes.append(pl["minutes"])
            calories.append(pl["calories"])
        return {
            "diets": sorted(diets),
            "minutes": {"min": min(minutes), "max": max(minutes)},
            "calories": {"min": min(calories), "max": max(calories)},
        }
    finally:
        client.close()


@mcp.prompt()
def plan_meal(diet: Diet, max_minutes: int = 30) -> str:
    """Plan a meal for a diet within a time budget: find a recipe, then show
    its nutrition and cooking steps. Exposed as a reusable slash command.

    diet must be one of: vegetarian, vegan, gluten-free, low-carb,
    very-low-carbs, low-fat, low-sodium, low-cholesterol, low-calorie,
    low-saturated-fat, low-protein, high-protein, healthy, diabetic.
    """
    return (
        f"Plan a meal for a {diet} diet that takes at most {max_minutes} minutes.\n"
        f"1. Call filter_recipes(diet='{diet}', max_minutes={max_minutes}) to find options.\n"
        f"2. Pick the best option and call calculate_nutrition on its name.\n"
        f"3. Call get_recipe_steps on the same recipe.\n"
        f"Then present the recipe name, a short nutrition summary, and the steps."
    )


if __name__ == "__main__":
    mcp.run()
