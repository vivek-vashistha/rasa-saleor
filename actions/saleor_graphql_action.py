import os
import json
import re
import logging
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType

from .kg_client import KGClient

load_dotenv()

log = logging.getLogger(__name__)

# Configuration from environment variables
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8002/chat_bot")
CHANNEL_SLUG = os.getenv("CHANNEL_SLUG", "default-channel")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEV_VERBOSE = os.getenv("DEV_VERBOSE_ERRORS", "0") == "1"


def call_saleor_api(
    question: str,
    kg_products: Optional[List[str]] = None,
    kg_response: Optional[Any] = None,
    session_id: str = "rasa_bot",
    channel: str = CHANNEL_SLUG
) -> Dict[str, Any]:
    """
    Call the Saleor API endpoint with the given parameters.
    
    Args:
        question: The user's question or query
        kg_products: List of product names from knowledge graph
        kg_response: Raw response from knowledge graph
        session_id: Session ID for conversation tracking
        channel: Channel slug for the request
        
    Returns:
        Dict containing the API response with 'answer' and 'queries' keys
    """
    # Prepare the request data according to the API's expected format
    data = {
        "question": question,
        "session_id": session_id,
        "channel_slug": channel,
        "additional_details": ""  # Required by API but can be empty
    }
    
    # Add KG products if available - ensure it's a list of strings
    if kg_products:
        if isinstance(kg_products, str):
            try:
                kg_products = json.loads(kg_products)
            except json.JSONDecodeError:
                kg_products = [kg_products]  # Convert single string to list
        
        if isinstance(kg_products, list):
            data["kg_products"] = json.dumps(kg_products)
            log.info(f"Sending kg_products: {data['kg_products']}")
    
    # Add KG response if available - pass through as is
    if kg_response is not None:
        if isinstance(kg_response, (dict, list)):
            data["kg_response"] = json.dumps(kg_response)
        else:
            data["kg_response"] = str(kg_response)
        log.info(f"Sending kg_response: {data['kg_response']}")
    
    log.info(f"Calling API at {BACKEND_URL} with data: {json.dumps(data, indent=2)}")
    
    try:
        # Make the API request with a reasonable timeout
        response = requests.post(
            BACKEND_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        
        log.info(f"API response status: {response.status_code}")
        
        # Try to parse JSON response
        try:
            response_data = response.json()
            log.info(f"API response data: {json.dumps(response_data, indent=2)}")
            
            # Ensure the response has the expected format
            if not isinstance(response_data, dict):
                raise ValueError("Unexpected response format: not a JSON object")
            
            if response_data and isinstance(response_data, dict):
                data = response_data.get("data", {})
                answer_block = data.get("answer", "")
            
            # Extract the "Final response" part if present
            final_answer = None
            if "### Final response" in answer_block:
                final_answer = answer_block.split("### Final response", 1)[1].strip()

            if final_answer:
                print(f"Sending answer to user:\n{final_answer}")
            else:
                print("Full answer block:\n", answer_block)
                
            return {
                "answer": final_answer,
                "queries": question
            }
            
        except json.JSONDecodeError:
            error_msg = f"Failed to parse API response as JSON: {response.text[:500]}"
            log.error(error_msg)
            return {
                "answer": "I received an invalid response from the product catalog.",
                "error": error_msg
            }
            
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        log.error(error_msg)
        if hasattr(e, 'response') and e.response is not None:
            log.error(f"Response status: {e.response.status_code}")
            log.error(f"Response text: {e.response.text}")
        
        return {
            "answer": "I'm having trouble connecting to the product catalog. Please try again later.",
            "error": error_msg
        }


class ActionSaleorGraphQL(Action):
    def name(self) -> str:
        return "action_saleor_graphql"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> List[EventType]:
        try:
            # Get slots and message text
            product = (tracker.get_slot("product_name") or "").strip()
            qtype = (tracker.get_slot("saleor_question_type") or "").strip()
            channel = (tracker.get_slot("channel_slug") or CHANNEL_SLUG).strip()
            user_id = (tracker.get_slot("user_identifier") or "").strip()
            user_text = (tracker.latest_message.get("text") or "").strip()
            kg_products = tracker.get_slot("kg_products") or []
            
            log.info(f"Processing request - product: {product}, qtype: {qtype}, channel: {channel}, "
                    f"user_id: {user_id}, user_text: {user_text}")
            
            # If user asked for prices/availability but didn't name products explicitly,
            # fall back to KG-derived products from the previous turn.
            wants_prices = bool(re.search(r"\b(price|prices|cost|buy|purchase|available|availability|in stock)\b", user_text, re.I))
            wants_avail = bool(re.search(r"\b(available|availability|in stock|stock|purchase|buy)\b", user_text, re.I))
            log.info(f"Detected - wants_prices: {wants_prices}, wants_avail: {wants_avail}")

            # If qtype not provided by NLU/form, infer from text
            if not qtype:
                if wants_prices and wants_avail:
                    qtype = "product_price_and_availability"
                elif wants_prices:
                    qtype = "product_pricing"
                elif wants_avail:
                    qtype = "product_availability"
                log.info(f"Inferred qtype: {qtype}")

            # Build the question based on available information
            if qtype and product:
                question = f"{qtype} for product '{product}' in channel '{channel}'."
            elif product:
                if qtype == "product_pricing":
                    question = f"Pricing only for '{product}' in channel '{channel}'. Return currency and clearly name each product."
                elif qtype == "product_availability":
                    question = f"Availability/publication only for '{product}' in channel '{channel}'."
                else:
                    question = f"Product info (price & availability) for '{product}' in channel '{channel}'. Return currency and clearly name each product."
            elif qtype == "user_info" and user_id:
                question = f"User information for '{user_id}' (orders, addresses, availability to purchase)."
            else:
                question = user_text or "Answer the user's question using the product catalog."

            log.info(f"Formatted question: {question}")
            
            # Use the original user text as the main question
            question = user_text or "Answer the user's question using the product catalog."

            log.info(f"Formatted question: {question}")
            
            # Get KG response from tracker if available
            # kg_response_slot = tracker.get_slot("kg_response")
            kg_response_slot = tracker.get_slot("kg_message")
            kg_response = None
            
            if kg_response_slot:
                log.info(f"Found kg_response_slot: {kg_response_slot}")
                # Try to parse as JSON first, if it fails keep as string
                if isinstance(kg_response_slot, str):
                    try:
                        # If it's a string, try to parse as JSON
                        kg_response = json.loads(kg_response_slot)
                        log.info("Successfully parsed kg_response as JSON")
                    except json.JSONDecodeError:
                        # If not valid JSON, use as is
                        kg_response = kg_response_slot
                        log.info("Using kg_response as string")
                else:
                    # If not a string, use as is
                    kg_response = kg_response_slot
                    log.info("Using kg_response as is")
            
            log.info(f"Calling API with kg_products: {kg_products}, kg_response: {kg_response is not None}")
            
            # Log the data being sent to the API
            log.info(f"Sending to API - Question: {question}")
            log.info(f"Sending to API - KG Products: {kg_products}")
            log.info(f"Sending to API - KG Response: {kg_response}")
            
            # Call the API with the question and KG products
            response = call_saleor_api(
                question=question,
                kg_products=kg_products,
                kg_response=kg_response,
                session_id=tracker.sender_id,
                channel=channel
            )

            log.info(f"API response: {json.dumps(response, indent=2) if isinstance(response, dict) else response}")

            # Handle the response
            if response and isinstance(response, dict):
                
                if "answer" in response:
                    answer = response["answer"]
                    log.info(f"Sending answer to user: {answer}")
                    dispatcher.utter_message(text=answer)
                    return [
                        SlotSet("saleor_last_answer", answer),
                        SlotSet("saleor_last_queries", json.dumps(response.get("queries", []))),
                        SlotSet("channel_slug", channel),
                    ]
                elif "error" in response:
                    error_msg = f"API error: {response['error']}"
            else:
                error_msg = f"Unexpected API response format: {type(response)}"
            
            error_msg = error_msg or "No answer returned from the API"
            log.error(error_msg)
            raise Exception(error_msg)

        except Exception as e:
            error_msg = str(e)
            log.exception(f"Saleor API action failed: {error_msg}")
            
            # Provide a more user-friendly error message
            if "ConnectionError" in error_msg:
                user_msg = "I'm having trouble connecting to the product catalog. Please check if the backend service is running."
            elif "Timeout" in error_msg:
                user_msg = "The request to the product catalog timed out. Please try again in a moment."
            elif "No answer" in error_msg:
                user_msg = "I couldn't get a response from the product catalog. The service might be temporarily unavailable."
            else:
                user_msg = "I encountered an error while processing your request."
                if DEV_VERBOSE:
                    user_msg += f" Technical details: {error_msg}"
            
            dispatcher.utter_message(text=user_msg)
            return [
                SlotSet("saleor_last_error", error_msg),
                SlotSet("saleor_last_queries", tracker.get_slot("saleor_last_queries") or "[]"),
            ]
