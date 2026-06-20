import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from rag.config import LLM_BASE_URL, LLM_MODEL


mcp_client = MultiServerMCPClient(
    {
        "recipes": {
            "command": "uv",
            "args": ["run", "python", "-m", "mcp_server.server"],
            "transport": "stdio",
        }
    }
)


ROLE = (
    "You are a recipe assistant backed by a recipe database. You answer using "
    "tools only, never from memory."
)
TOOLS_GUIDE = (
    "Tools:\n"
    "- search_recipes — open-ended requests (dish ideas, ingredients, a vibe).\n"
    "- filter_recipes — hard constraints (a diet, a max cook time, a calorie ceiling).\n"
    "- calculate_nutrition — a recipe's nutrition; pass the exact name from search/filter.\n"
    "- get_recipe_steps — a recipe's cooking instructions; pass the exact name too.\n"
    "Note: nutrition values except calories are % of daily value, not grams."
)

REASONING_PROMPT = """Before each action, think step-by-step:
1. What information do I currently have?
2. What information do I still need?
3. Which tool will help me get that information?
4. What parameters should I use?"""

ERROR_PROMPT = """If a tool returns an error or unexpected result:
- Think about why it might have failed
- Consider alternative approaches
- Try different parameters or a different tool
- Never make up information if tools fail"""

FOCUSED_PROMPT = """Guidelines:
- Always check prerequisites before main actions
- Minimize tool calls by planning ahead
- Combine related queries when possible
- Stop when you have sufficient information"""

FORMAT_PROMPT = """For your final answer:
- Summarize all key findings
- Be specific with numbers and names
- Acknowledge any limitations or assumptions
- Provide clear next steps if applicable"""

SYSTEM_PROMPT = SystemMessage(
    "\n\n".join(
        [
            ROLE,
            TOOLS_GUIDE,
            REASONING_PROMPT,
            ERROR_PROMPT,
            FOCUSED_PROMPT,
            FORMAT_PROMPT,
        ]
    )
)


async def load_tools():
    return await mcp_client.get_tools()


def make_llm(tools):
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key="not-needed",
        model=LLM_MODEL,
        temperature=0.3,
    )
    return llm.bind_tools(tools)


async def make_graph():
    tools = await load_tools()
    llm = make_llm(tools)

    def agent_node(state: MessagesState):
        return {"messages": [llm.invoke([SYSTEM_PROMPT] + state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


async def main():
    app = await make_graph()
    result = await app.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Find me a vegan dinner under 30 minutes, then show its nutrition.",
                }
            ]
        }
    )
    for m in result["messages"]:
        m.pretty_print()


asyncio.run(main())
