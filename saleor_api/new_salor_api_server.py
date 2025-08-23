#!/usr/bin/env python3
"""
Simple API server for the new Saleor LangGraph (`new_salor_graph.py`).

This mirrors the style of `saleor_api_server.py` but drives the graph defined in
`new_salor_graph.py`. It exposes:

- POST /chat_bot        -> executes one turn with the graph
- POST /clear_chat_bot  -> clears the in-memory session state
- GET  /health          -> basic health check

Run:
  uvicorn new_salor_api_server:app --host 0.0.0.0 --port 8002 --log-level info

Environment:
- OPENAI_API_KEY must be set
- SALEOR_* variables are read by new_salor_graph.py as needed

Notes:
- Session is maintained in-memory by `SESSIONS` using the `session_id` form field.
- `kg_products` and `kg_response` are optional and can be JSON or text.
"""

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import logging
import time
import gc
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Import graph helpers
from langchain_core.messages import AIMessage
from new_salor_graph import run_structured, MessagesState  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="New Saleor Graph API Server",
    description="API wrapper for new_salor_graph LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}


def create_api_response(status, success_count=None, failed_count=None, data=None, error=None, message=None, file_source=None, file_name=None):
    """
    Create a response to be sent to the API (mirrors saleor_api_server.py style).
    """
    response = {"status": status}

    if data is not None:
        response["data"] = data

    if error is not None:
        response["error"] = error
    
    if success_count is not None:
        response['success_count'] = success_count
        response['failed_count'] = failed_count
    
    if message is not None:
        response['message'] = message

    if file_source is not None:
        response['file_source'] = file_source

    if file_name is not None:
        response['file_name'] = file_name
        
    return response


def formatted_time(dt):
    """Format datetime for logging"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# In-memory session store for MessagesState
SESSIONS: Dict[str, MessagesState] = {}


def _parse_optional_json(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


@app.post("/chat_bot")
async def chat_bot(
    # Keep compatibility with the form signature in saleor_api_server.py
    uri: str = Form(None),
    model: str = Form(None), 
    userName: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    question: str = Form(None),
    document_names: str = Form(None),
    session_id: str = Form(None),
    mode: str = Form(None),
    email: str = Form(None),
    user_id: str = Form(None),
    # Optional structured fields for this graph
    additional_details: str = Form(None),
    kg_products: str = Form(None),
    kg_response: str = Form(None),
):
    """
    One-turn execution against the new Saleor graph.

    - `question` maps to "Users query" in the graph's structured input.
    - `additional_details`, `kg_products`, and `kg_response` are optional.
    - Maintains per-session message history in-memory.
    """
    logger.info(f"New Saleor Graph chat_bot called at {datetime.now()} | session={session_id}")
    start_time = time.time()

    try:
        # Minimal environment check
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is required")

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        # Prepare session state
        sid = session_id or "default"
        state = SESSIONS.get(sid) or {"messages": []}

        # Parse optional inputs
        parsed_additional_details = additional_details.strip() if additional_details else None
        parsed_kg_products = _parse_optional_json(kg_products)
        parsed_kg_response = _parse_optional_json(kg_response)

        # Execute graph on a thread (non-blocking the event loop)
        new_state: MessagesState = await asyncio.to_thread(
            run_structured,
            users_query=question,
            additional_details=parsed_additional_details,
            kg_products=parsed_kg_products,
            kg_response=parsed_kg_response,
            state=state,
        )

        # Save back session
        SESSIONS[sid] = new_state

        # Extract last AI message
        ai_msg = next(m for m in reversed(new_state["messages"]) if isinstance(m, AIMessage))

        # Calc timing
        total_call_time = time.time() - start_time
        logger.info(f"New Saleor Graph response time: {total_call_time:.2f}s | session={sid}")

        # Build result payload
        result = {
            "answer": ai_msg.content,
            "session_id": sid,
            "messages_count": len(new_state["messages"]),
            "info": {
                "response_time": round(total_call_time, 2),
                "model": model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            },
        }

        # Log like the sample server
        json_obj = {
            'api_name': 'new_saleor_graph_chat_bot',
            'db_url': uri,
            'userName': userName,
            'database': database,
            'question': question,
            'document_names': document_names,
            'session_id': sid,
            'mode': mode,
            'logging_time': formatted_time(datetime.now(timezone.utc)),
            'elapsed_api_time': f'{total_call_time:.2f}',
            'email': email,
            'user_id': user_id
        }
        logger.info(f"API call logged: {json_obj}")

        return create_api_response('Success', data=result)

    except Exception as e:
        job_status = "Failed"
        message = "Unable to get chat response from new Saleor graph"
        error_message = str(e)
        logger.exception(f'Exception in new Saleor graph chat bot: {error_message}')
        return create_api_response(job_status, message=message, error=error_message, data=mode)

    finally:
        gc.collect()


@app.post("/clear_chat_bot")
async def clear_chat_bot(session_id: str = Form(...)):
    """
    Clear chat history for a session (in-memory state).
    """
    try:
        logger.info(f"Clear chat bot called for session: {session_id}")
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        return create_api_response('Success', message=f"Chat history cleared for session {session_id}")
    except Exception as e:
        job_status = "Failed"
        message = "Unable to clear chat history"
        error_message = str(e)
        logger.exception(f'Exception in clear chat bot: {error_message}')
        return create_api_response(job_status, message=message, error=error_message)


if __name__ == "__main__":
    # Load environment variables from .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")

    # Run the server
    uvicorn.run(
        app,
        host="localhost",
        port=8002,
        log_level="info"
    )
