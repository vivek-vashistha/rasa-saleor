# actions/saleor_graphql_action.py
import os, json, re, logging
from dotenv import load_dotenv
load_dotenv()

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_community.tools.graphql.tool import BaseGraphQLTool

log = logging.getLogger(__name__)

SALEOR_ENDPOINT = os.getenv("SALEOR_ENDPOINT", "")
SALEOR_TOKEN    = os.getenv("SALEOR_TOKEN", "")
CHANNEL_SLUG    = os.getenv("CHANNEL_SLUG", "default-channel")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMP     = float(os.getenv("OPENAI_TEMPERATURE", "0"))
DEV_VERBOSE     = os.getenv("DEV_VERBOSE_ERRORS", "0") == "1"

headers = {"Authorization": f"Bearer {SALEOR_TOKEN}"} if SALEOR_TOKEN else None

# Build two wrappers: prefer NO schema; keep a fallback WITH schema.
graphql_wrapper_without_schema = GraphQLAPIWrapper(
    graphql_endpoint=SALEOR_ENDPOINT,
    custom_headers=headers,
    fetch_schema_from_transport=False,
)
graphql_wrapper_with_schema = GraphQLAPIWrapper(
    graphql_endpoint=SALEOR_ENDPOINT,
    custom_headers=headers,
    fetch_schema_from_transport=True,
)

saleor_tool_description = (
    "Input is a valid Saleor GraphQL query/mutation string.\n"
    f"- Channel context is \"{CHANNEL_SLUG}\". For pricing/availability/publication, use this channel.\n"
    "- NEVER add arguments to 'channelListings'. Instead fetch the list and SELECT the item whose channel.slug "
    f'equals \"{CHANNEL_SLUG}\".\n'
    "- When searching, prefer: products(channel: \"<slug>\", filter: { search: \"keywords\" }, first: 10) { ... }\n"
    "- For price ranges, use: pricing { priceRange { start { gross { amount currency } } stop { gross { amount currency } } } }\n"
    "- Do not invent fields; stick to the schema."
)

graphql_tool_without_schema = BaseGraphQLTool(
    graphql_wrapper=graphql_wrapper_without_schema,
    description=saleor_tool_description,
)
graphql_tool_with_schema = BaseGraphQLTool(
    graphql_wrapper=graphql_wrapper_with_schema,
    description=saleor_tool_description,
)

try:
    graphql_tool_without_schema.name = "query_graphql"
    graphql_tool_with_schema.name = "query_graphql"
except Exception:
    pass

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMP).bind_tools(
    [graphql_tool_without_schema, graphql_tool_with_schema]
)

SYSTEM_PROMPT = f"""You are a Saleor shopping copilot for an online retailer.
Use exactly ONE tool: query_graphql (you may call it multiple times).

Follow these rules:
- Channel context is "{CHANNEL_SLUG}". For pricing/availability/publication, use this channel.
- NEVER put arguments on 'channelListings'. Instead fetch:
    channelListings {{
      channel {{ slug }}
      isPublished
      visibleInListings
      availableForPurchaseAt
      pricing {{
        priceRange {{
          start {{ gross {{ amount currency }} }}
          stop  {{ gross {{ amount currency }} }}
        }}
      }}
    }}
  and use the entry where channel.slug == "{CHANNEL_SLUG}".
- For product search, ALWAYS paginate:
    products(channel: "{CHANNEL_SLUG}", filter: {{ search: "<keywords>" }}, first: 10) {{ ... }}
- State the currency in your price answer. Do not invent fields.
Return a short, helpful answer after you finish tool calls.
"""

# --- Query sanitizers for common schema pitfalls --------------------------------
def _sanitize_query(q: str) -> str:
    """Fix common issues before sending to Saleor."""
    if not isinstance(q, str):
        return q
    fixed = q

    # 1) Remove illegal args from channelListings(...)
    fixed = re.sub(r'(channelListings)\s*\([^)]*\)', r'\1', fixed)

    # 2) priceRange uses 'stop', not 'end'
    fixed = re.sub(r'\bend\s*\{', 'stop {', fixed)

    # 3) Ensure products(...) has first: or last:
    # If products( ... ) lacks 'first:' or 'last:', inject first: 10 at the beginning.
    def add_first(match):
        inner = match.group(1)
        if re.search(r'\b(first|last)\s*:', inner):
            return f'products({inner})'
        inner = inner.strip()
        return 'products(first: 10' + (', ' + inner if inner else '') + ')'
    fixed = re.sub(r'products\s*\(\s*([^)]+)\)', add_first, fixed)

    # Tidy ', )' -> ')'
    fixed = re.sub(r',\s*\)', ')', fixed)
    return fixed

def _invoke_graphql(tool_input: str):
    """Prefer no-schema transport (avoids introspection error); if it fails, try with-schema."""
    query = _sanitize_query(tool_input) if isinstance(tool_input, str) else tool_input
    try:
        return graphql_tool_without_schema.invoke(query)
    except Exception as e:
        # As a fallback only, try the schema client (may log the includeDeprecated warning)
        try:
            return graphql_tool_with_schema.invoke(query)
        except Exception as e2:
            # Surface the original error context for logs
            # raise e2
            # Surface the full context for logs & (optionally) the user
            raise RuntimeError(f"GraphQL failed. No-schema error: {repr(e)} | With-schema error: {repr(e2)}")

def _tool_loop(user_task: str, max_iters: int = 4) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_task)]
    tool_queries, last_ai = [], None

    for _ in range(max_iters):
        last_ai = llm.invoke(messages)
        tool_calls = getattr(last_ai, "tool_calls", None)
        if not tool_calls:
            break

        # Sanitize tool_calls so they are JSON-serializable when the messages are resent
        sanitized_calls = []
        for tc in tool_calls:
            name = getattr(tc, "name", None) or (isinstance(tc, dict) and tc.get("name"))
            call_id = getattr(tc, "id", None) or (isinstance(tc, dict) and tc.get("id"))
            args = getattr(tc, "args", None) or (isinstance(tc, dict) and tc.get("args")) or {}
            query_arg = args.get("query") if isinstance(args, dict) else None
            if not query_arg and isinstance(args, dict):
                query_arg = args.get("input")
            if isinstance(query_arg, str):
                safe_args = {"query": _sanitize_query(query_arg)}
            elif isinstance(args, dict):
                # Keep only JSON-serializable primitives, and ensure we have a query key
                safe_args = {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool)) or v is None}
                if "query" not in safe_args:
                    safe_args["query"] = str(query_arg) if query_arg is not None else str(args)
            else:
                safe_args = {"query": str(args)}
            sanitized_calls.append({"name": name, "id": call_id, "args": safe_args})

        # Append sanitized assistant message (with sanitized tool calls)
        messages.append(AIMessage(content=last_ai.content, tool_calls=sanitized_calls))

        # Now execute tools and append their results
        for tc in tool_calls:
            args = getattr(tc, "args", None) or (isinstance(tc, dict) and tc.get("args")) or {}
            # Prefer the raw GraphQL string from the tool call if present
            query_arg = args.get("query") or args.get("input")
            tool_input = query_arg if isinstance(query_arg, str) else args
            res = _invoke_graphql(tool_input)
            # Record only a serializable representation of the query to avoid non-JSON-serializable objects
            if isinstance(query_arg, str):
                tool_queries.append(_sanitize_query(query_arg))
            elif isinstance(tool_input, str):
                tool_queries.append(_sanitize_query(tool_input))
            else:
                tool_queries.append(str(args))
            call_id = getattr(tc, "id", None) or (isinstance(tc, dict) and tc.get("id"))
            messages.append(ToolMessage(content=str(res), tool_call_id=call_id))

    answer_text = last_ai.content if last_ai else "Sorry, I couldn't generate an answer."
    return {"answer": answer_text, "queries": tool_queries}

class ActionSaleorGraphQL(Action):
    def name(self) -> str:
        return "action_saleor_graphql"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list[EventType]:
        product = (tracker.get_slot("product_name") or "").strip()
        qtype   = (tracker.get_slot("saleor_question_type") or "").strip()
        channel = (tracker.get_slot("channel_slug") or CHANNEL_SLUG).strip()
        user_id = (tracker.get_slot("user_identifier") or "").strip()
        user_text = (tracker.latest_message.get("text") or "").strip()
        kg_products = tracker.get_slot("kg_products") or []

        # If user asked for prices/availability but didn't name products explicitly,
        # fall back to KG-derived products from the previous turn.
        wants_prices = bool(re.search(r"\b(price|prices|cost|buy|purchase|available|availability|in stock)\b", user_text, re.I))
        wants_avail  = bool(re.search(r"\b(available|availability|in stock|stock|purchase|buy)\b", user_text, re.I))

        # If qtype not provided by NLU/form, infer from text
        if not qtype:
            if wants_prices and wants_avail:
                qtype = "product_price_and_availability"
            elif wants_prices:
                qtype = "product_pricing"
            elif wants_avail:
                qtype = "product_availability"

        # Build task string depending on slots
        if qtype and product:
            task = f"{qtype} for product '{product}' in channel '{channel}'."
        elif product:
            if qtype == "product_pricing":
                task = f"Pricing only for '{product}' in channel '{channel}'. Return currency and clearly name each product."
            elif qtype == "product_availability":
                task = f"Availability/publication only for '{product}' in channel '{channel}'."
            else:
                task = f"Product info (price & availability) for '{product}' in channel '{channel}'. Return currency and clearly name each product."
        elif qtype == "user_info" and user_id:
            task = f"User information for '{user_id}' (orders, addresses, availability to purchase)."
        else:
            task = user_text or "Answer the user's question using Saleor GraphQL."

        try:
            # Case 1: multiple KG products, no explicit product slot
            if not product and kg_products and (wants_prices or not qtype):
                results, queries = [], []
                for prod in kg_products:
                    task = f"{qtype or 'product_info'} for product '{prod}' in channel '{channel}'."
                    res = _tool_loop(task)
                    results.append(res["answer"])
                    queries.extend(res["queries"])

                final_answer = "\n".join(results)
                dispatcher.utter_message(text=final_answer)
                return [
                    SlotSet("saleor_last_answer", final_answer),
                    SlotSet("saleor_last_queries", json.dumps(queries)),
                    SlotSet("channel_slug", channel),
                ]

            # Case 2: single product or general query
            else:
                result = _tool_loop(task)
                dispatcher.utter_message(text=result["answer"])
                return [
                    SlotSet("saleor_last_answer", result["answer"]),
                    SlotSet("saleor_last_queries", json.dumps(result["queries"])),
                    SlotSet("channel_slug", channel),
                ]

        except Exception as e:
            log.exception("Saleor GraphQL action failed: %s", e)
            err_msg = f"I couldn’t complete the catalog query."
            if DEV_VERBOSE:
                err_msg += f" Details: {e}"
            dispatcher.utter_message(text=err_msg)
            return [
                SlotSet("saleor_last_error", str(e)),
                SlotSet("saleor_last_queries", tracker.get_slot("saleor_last_queries")),
            ]
