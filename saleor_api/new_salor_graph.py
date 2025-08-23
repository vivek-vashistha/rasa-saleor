import os
import json
from typing import Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode

# Use GraphQL wrapper + tool from langchain_community
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_community.tools.graphql.tool import BaseGraphQLTool

import logging

class LengthFilter(logging.Filter):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def filter(self, record):
        # Only allow log messages up to max_length characters
        return len(record.getMessage()) <= self.max_length

# Get the gql transport logger
logger = logging.getLogger("gql.transport.requests")

# Set level to INFO so short messages still show
logger.setLevel(logging.INFO)

# Add our length filter
logger.addFilter(LengthFilter(max_length=2500))

# ---------------------------------
# Config
# ---------------------------------
SALEOR_ENDPOINT = os.getenv("SALEOR_ENDPOINT", "https://store-gqt4azfa.saleor.cloud/graphql/")
SALEOR_TOKEN    = os.getenv("SALEOR_TOKEN", "REPLACE_WITH_YOUR_API_TOKEN")
CHANNEL_SLUG    = os.getenv("CHANNEL_SLUG", "default-channel")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


EXTERNAL_KG_CONTEXT: Any = None
EXTERNAL_KG_MESSAGE: Optional[str] = None

def set_external_kg_context(ctx: Any) -> None:
    """
    Receives the KG context. Normalizes and stores the human-readable 'message'
    so the LLM can use it as background context.
    """
    global EXTERNAL_KG_CONTEXT, EXTERNAL_KG_MESSAGE
    EXTERNAL_KG_CONTEXT = ctx
    EXTERNAL_KG_MESSAGE = None

    # Extract the most useful human text
    try:
        msg = None
        if isinstance(ctx, dict):
            # Prefer data.message, then fall back to top-level message
            data = ctx.get("data") or {}
            msg = data.get("message") or ctx.get("message")
        elif isinstance(ctx, str):
            msg = ctx
        if msg is not None:
            EXTERNAL_KG_MESSAGE = str(msg).strip()
    except Exception:
        pass

    # Log a compact receipt
    try:
        preview = None
        if isinstance(ctx, dict):
            preview = {
                "status": ctx.get("status"),
                "has_message": bool(EXTERNAL_KG_MESSAGE),
                "message_preview": (EXTERNAL_KG_MESSAGE[:240] + "…") if EXTERNAL_KG_MESSAGE and len(EXTERNAL_KG_MESSAGE) > 240 else EXTERNAL_KG_MESSAGE,
            }
        else:
            preview = str(ctx)[:240]
        logger.info(f"SALEOR_TOOL | kg_context_received | {json.dumps(preview)}")
    except Exception:
        logger.info("SALEOR_TOOL | kg_context_received | <unserializable>")

def get_external_kg_message() -> Optional[str]:
    """Returns the normalized KG message (if any)."""
    return EXTERNAL_KG_MESSAGE

# ---------------------------------
# GraphQL tool (UNCHANGED DESCRIPTION)
# ---------------------------------
headers = {"Authorization": f"Bearer {SALEOR_TOKEN}"} if SALEOR_TOKEN and SALEOR_TOKEN != "REPLACE_WITH_YOUR_API_TOKEN" else None
graphql_wrapper = GraphQLAPIWrapper(
    graphql_endpoint=SALEOR_ENDPOINT,
    custom_headers=headers,
    fetch_schema_from_transport=True,
)

graphql_tool = BaseGraphQLTool(
    graphql_wrapper=graphql_wrapper,
    # IMPORTANT: Per user request, keep the same description
    description=(
        "Input is a valid Saleor GraphQL query/mutation string. "
        "For ANY availability/publication/pricing, ALWAYS pass the channel argument "
        f'and call product(..., channel: "{CHANNEL_SLUG}"). '
        "Never use isAvailable without a channel. Prefer Product.channelListings "
        "for the target channel (isPublished, availableForPurchaseAt, visibleInListings)."
    )
)

TOOLS = [graphql_tool]

# ---------------------------------
# Structured input helper
# ---------------------------------
@dataclass
class StructuredInput:
    users_query: str
    additional_details: Optional[str] = None
    kg_products: Optional[Any] = None
    kg_response: Optional[Any] = None


def _render_structured_input(si: StructuredInput) -> str:
    def _fmt(obj: Any) -> str:
        if obj is None:
            return "N/A"
        try:
            if isinstance(obj, str):
                return obj
            return json.dumps(obj, ensure_ascii=False)[:4000]
        except Exception:
            return str(obj)[:4000]

    return (
        "# INPUTS\n"
        f"Users query: {si.users_query}\n"
        f"Additional details: {_fmt(si.additional_details)}\n"
        f"KG products: {_fmt(si.kg_products)}\n"
        f"KG response: {_fmt(si.kg_response)}\n"
    )

# ---------------------------------
# System prompt: simplified flow + strict output
# ---------------------------------
SYSTEM_PROMPT = f"""
You are a Saleor shopping copilot for online retailers. You have exactly ONE tool: query_graphql. You may call it multiple times per turn.

HOW TO USE THE INPUT
- Each user turn is provided in this structure: "Users query", "Additional details", "KG products" (optional), and "KG response" (optional).
- Treat the two KG sections as helpful background only; ALWAYS verify details (esp. price/currency and channel data) via Saleor GraphQL.

WHAT TO DO (simplified flow)
1) Read the Inputs. Combine "Users query" + "Additional details" (+KG if present).
2) Decide what is missing to fully answer.
3) Generate one or more sub-queries (GraphQL) to fetch anything missing and/or final data.
4) Call the GraphQL tool for each sub-query. You MAY call it multiple times.
5) Summarize tool outputs and return the final answer.

OUTPUT FORMAT (must match EXACTLY; no extra sections)
### Inputs
- users_query: <verbatim from input>
- additional_details: <short>
- kg_products: <short or N/A>
- kg_response: <short or N/A>

### Generated queries
<One bullet per sub-query. For each, state purpose and the GraphQL operation name or a short label. Do NOT reveal chain-of-thought.>

### Tool call responses
<One short bullet block per sub-query: 1–3 bullets summarizing the key data returned (product names/IDs, availability, currency and price ranges, etc.).>

### Final response
<Concise, user-facing answer that directly addresses the original request. Include currency/channel where relevant.>
"""

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0).bind_tools(TOOLS)

# ---------------------------------
# Minimal graph
# ---------------------------------

def _sanitize_for_llm(messages: List[Any]) -> List[Any]:
    clean: List[Any] = []
    pending_tools_allowed = False
    for m in messages:
        if isinstance(m, AIMessage):
            clean.append(m)
            pending_tools_allowed = bool(getattr(m, "tool_calls", None))
        elif isinstance(m, ToolMessage):
            if pending_tools_allowed:
                clean.append(m)
        else:
            clean.append(m)
            pending_tools_allowed = False
    return clean


def call_llm(state: MessagesState):
    # Clean the chat history as you already do
    history = _sanitize_for_llm(state["messages"])

    # Base system prompt
    sys_prompt = SYSTEM_PROMPT

    # If we have a KG message, feed it as background
    kg_msg = get_external_kg_message()
    if kg_msg:
        sys_prompt += (
            "\n\n# Background from Knowledge Graph\n"
            f"{kg_msg}\n\n"
            "Use this as helpful context. Always verify details (especially price/currency) via Saleor GraphQL. "
            "If KG lacks USD or specific currency, fetch actual prices from Saleor."
        )
        try:
            logger.info(f"\n\nSALEOR_TOOL | kg_message_injected | sys_prompt length={len(sys_prompt)}\n\n")
        except Exception:
            pass

    msgs = [SystemMessage(content=sys_prompt)] + history
    ai = llm.invoke(msgs)
    return {"messages": [ai]}


tool_node = ToolNode(TOOLS)


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


graph = StateGraph(MessagesState)
graph.add_node("llm", call_llm)
graph.add_node("tool_exec", tool_node)
graph.add_conditional_edges("llm", should_continue, {"tools": "tool_exec", END: END})
graph.add_edge("tool_exec", "llm")
graph.set_entry_point("llm")
app = graph.compile()

# ---------------------------------
# Convenience helpers for your desired input structure
# ---------------------------------

def make_structured_message(
    users_query: str,
    additional_details: Optional[str] = None,
    kg_products: Optional[Any] = None,
    kg_response: Optional[Any] = None,
) -> HumanMessage:
    """Build a HumanMessage following your desired four-field input."""
    si = StructuredInput(
        users_query=users_query,
        additional_details=additional_details,
        kg_products=kg_products,
        kg_response=kg_response,
    )

    # Also push a compact KG note into the background channel (optional)
    if kg_products or kg_response:
        try:
            combined_msg = "".join([
                (kg_response if isinstance(kg_response, str) else json.dumps(kg_response, ensure_ascii=False)) if kg_response is not None else "",
                "\n",
                (kg_products if isinstance(kg_products, str) else json.dumps(kg_products, ensure_ascii=False)) if kg_products is not None else "",
            ]).strip()
            if combined_msg:
                set_external_kg_context({"message": combined_msg})
        except Exception:
            pass

    return HumanMessage(content=_render_structured_input(si))


def run_structured(
    users_query: str,
    additional_details: Optional[str] = None,
    kg_products: Optional[Any] = None,
    kg_response: Optional[Any] = None,
    state: Optional[MessagesState] = None,
) -> MessagesState:
    """
    One-call helper that:
      - formats your four input fields
      - invokes the compiled graph
      - returns the updated state
    """
    if state is None:
        state = {"messages": []}
    state["messages"].append(
        make_structured_message(
            users_query=users_query,
            additional_details=additional_details,
            kg_products=kg_products,
            kg_response=kg_response,
        )
    )
    return app.invoke(state)


# ---------------------------------
# Simple structured REPL (optional)
# ---------------------------------
if __name__ == "__main__":
    print("Saleor GraphQL Tool Agent (Simplified Flow) — type 'exit' to quit.\n")
    state: MessagesState = {"messages": []}
    while True:
        uq = input("Users query: ").strip()
        # uq = "can you suggest me some healthy drinks along with there prices."
        if uq.lower() == "exit":
            break
        ad = input("Additional details (optional): ").strip() or None
        kp = input("KG products (optional, JSON or text): ").strip()
        # kp = [["Carrot Juice","Banana Juice", "Bean Juice"]]
        kp_val = None
        if kp:
            try:
                kp_val = json.loads(kp)
            except Exception:
                kp_val = kp
        kr = input("KG response (optional, JSON or text): ").strip()
        # kr = [ "Here are some healthy drink options based on the provided context:\n\n1. **Carrot Juice**: Made from 100% pure, squeezed carrots, it offers the sweet, orange nectar of Mother Earth and helps improve eyesight naturally.\n2. **Banana Juice**: An exotic drink made from ripe bananas, packed with natural protein and the goodness of the tropical sun.\n3. **Bean Juice**: A health-conscious energy drink made from beans, prepared from allotment to bottle in under 8 hours.\n\nLet me know if you'd like more details about any of these!"]
        kr_val = None
        if kr:
            try:
                kr_val = json.loads(kr)
            except Exception:
                kr_val = kr

        state = run_structured(uq, ad, kp_val, kr_val, state)
        ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        print("\nAssistant:\n", ai.content, "\n")
