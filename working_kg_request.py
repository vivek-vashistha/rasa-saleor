import os, json, requests

# Set the required environment variable for testing
# os.environ["LLM_MODEL_CONFIG_openai_gpt_4o"] = "gpt-4o-2024-11-20,sk-your-openai-api-key-here"

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat_bot")

def ask_chatbot(
    uri, userName, password,
    database="neo4j",
    question="What's in our graph about <topic>?",
    model="openai_gpt_4o",
    mode="graph_vector_fulltext",
    session_id="test",
    document_names=None,                   # list[str] or None
):
    data = {
        "uri": uri,
        "userName": userName,
        "password": password,
        "database": database,
        "model": model,
        "question": question,
        "mode": mode,
        "session_id": session_id,
        "document_names": json.dumps(document_names or []),
    }
    r = requests.post(BACKEND_URL, data=data, timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    ask_chatbot(
        uri="neo4j+s://mkmm.databases.neo4j.io",
        userName="neo4j",
        password="kkkkk",
        question="What juices are there?",
    )
