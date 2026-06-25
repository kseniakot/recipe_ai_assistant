# Recipe Assistant — RAG + MCP + ReAct Agent

A domain-specific AI assistant for **cooking recipes** from Food.com that combines
Retrieval-Augmented Generation over a recipe corpus with tool-based actions
exposed through an MCP server, driven by a LangGraph ReAct agent.

## Domain

The assistant answers recipe questions from a knowledge base of ~500 recipes
(sampled from the [Food.com dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)).
Recipes are a good fit for **RAG + tools**: there is a large body of
self-contained documents to retrieve over (names, ingredients, steps), and each
recipe carries rich structured metadata — diet tags, cook time, and a full
nutrition breakdown. That split maps cleanly onto the two halves of the project:
fuzzy *"what can I cook with…"* questions are answered by semantic **retrieval**,
while hard constraints (*vegan*, *under 30 minutes*, *low-carb*) and derived facts
(*how many calories?*, *what are the steps?*) are answered by deterministic
**tools** over the metadata.

## Architecture

```
                    ┌──────────────────────────────────────────┐
   user ──▶ LangGraph ReAct agent  (local Llama-3.1-8B, LM Studio)
                    │   agent ⇄ tools loop                      │
                    └───────────────┬──────────────────────────┘
                                    │ langchain-mcp-adapters (stdio)
                    ┌───────────────▼──────────────────────────┐
                    │  MCP server (FastMCP)                     │
                    │   tools:  search_recipes, filter_recipes, │
                    │           calculate_nutrition, get_recipe_steps
                    │   resource: recipes://filters             │
                    │   prompt:   plan_meal                     │
                    └───────────────┬──────────────────────────┘
                       RAG pipeline  │
   query ▶ multi-query (LLM) ▶ hybrid search (dense+sparse, RRF) ▶ outer RRF
         ▶ cross-encoder rerank ▶ top-k recipes
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │  Qdrant  (BGE-M3 dense + sparse vectors)  │
                    └──────────────────────────────────────────┘
```

### RAG implementation

- **Documents:** ~500 recipes (Food.com), one document per recipe.
- **Indexing:** each recipe → `name + ingredients + description` text, embedded
  with **BGE-M3** (dense **and** sparse from one pass); metadata (tags, minutes,
  nutrition, steps) stored in the Qdrant payload. Per-document metadata
  extraction is used instead of chunking since recipes are short, self-contained
  documents.
- **Vector store:** **Qdrant** with named `dense` + `sparse` vectors and payload
  indexes on `tags`, `minutes`, `calories`.
- **Advanced techniques:**
  - **Multi-query** — the LLM rewrites the query into several phrasings.
  - **Hybrid search** — dense + sparse fused with **Reciprocal Rank Fusion**
    across the multi-query variants.
  - **Cross-encoder reranking** — `bge-reranker-v2-m3` re-scores candidates
    against the original query.

### MCP server (`mcp_server/server.py`)

- **Tools:** `search_recipes` (RAG), `filter_recipes` (metadata filter),
  `calculate_nutrition`, `get_recipe_steps`.
- **Resource:** `recipes://filters` — valid diets + cook-time / calorie ranges.
- **Prompt:** `plan_meal` — a parameterised slash-command template.
- Input validation, defensive error handling, `ctx` logging, and progress
  reporting on the long `search_recipes` call.

### Agent (`agent.py`)
- A hand-built **LangGraph ReAct** graph (`agent ⇄ tools` with a conditional
  edge), tools loaded from the MCP server via `langchain-mcp-adapters`.
- System prompt built from the four course prompting strategies (reasoning,
  error-handling, search-constraining, output-formatting) plus a domain guide.

## Setup

### Prerequisites

- Python ≥ 3.14, [`uv`](https://docs.astral.sh/uv/), Docker.
- [LM Studio](https://lmstudio.ai) with **Llama-3.1-8B-Instruct** loaded and the
  local server running on `http://localhost:1234`.

### 1. Install

```bash
uv sync
```

### 2. Start Qdrant

```bash
docker compose up -d        # Qdrant on :6333
```

### 3. Get the dataset

Download the Food.com dataset and place `RAW_recipes.csv` at `data/RAW_recipes.csv`
(e.g. via `kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")`).

### 4. Index the recipes into Qdrant

```bash
uv run python -m main        # samples 500 recipes, embeds, upserts
```

### 5. Run

```bash
# inspect the MCP server interactively
uv run mcp dev mcp_server/server.py

# the agent in the terminal
uv run python agent.py

# the agent in LangGraph Studio (visual trace)
uv run langgraph dev
```

Configuration (Qdrant URL, collection name, sample size, LLM endpoint/model)
is in `rag/config.py`.
