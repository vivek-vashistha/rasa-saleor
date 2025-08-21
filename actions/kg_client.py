# actions/kg_client.py
import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

def _endpoint_from(base: str) -> str:
    base = (base or "").rstrip("/")
    # if user already set /chat_bot, don't double-append
    return base if base.endswith("/chat_bot") else f"{base}/chat_bot"

class KGClient:
    """
    Minimal client for your Knowledge Graph server.

    It POSTs to {API_BASE_URL}/chat_bot with the same fields used in your
    working request script and returns the JSON response.
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        timeout: int = 60,
    ):
        base = api_base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
        self.endpoint = _endpoint_from(base)
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE")
        self.model = model or os.getenv("KG_MODEL")
        self.mode = mode or os.getenv("KG_MODE")
        self.timeout = timeout
        logger.info(f"KG endpoint: {self.endpoint}")

    def query(
        self,
        question: str,
        session_id: str = "default",
        document_names: Optional[List[str]] = None,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        data = {
            "uri": self.neo4j_uri,
            "userName": self.neo4j_username,
            "password": self.neo4j_password,
            "database": self.neo4j_database,
            "model": self.model,
            "question": question,
            "mode": self.mode,
            "session_id": session_id,
            "document_names": json.dumps(document_names or []),
            "email": email,
        }
        try:
            resp = requests.post(self.endpoint, data=data, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            # bubble up a helpful message for the action
            body = getattr(e, "response", None)
            body = getattr(body, "text", "") if body is not None else ""
            logger.exception("KG request failed: %s | body=%s", e, body)
            raise

    # ---------- Convenience extractors (purely optional) ----------

    @staticmethod
    def extract_products_from_message(message: str) -> List[str]:
        """
        Grab product names from a natural-language message, e.g.
        'The available juices are Carrot Juice, Apple Juice, Bean Juice, and Banana Juice.'
        """
        # naive list split after common patterns
        m = re.search(
            r"(?:are|include|includes|consist of|available(?:\s\w+)?\s(?:are|:))\s(.+)$",
            message.strip(),
            flags=re.IGNORECASE,
        )
        if not m:
            return []
        tail = m.group(1)
        # split by commas and 'and'
        parts = re.split(r",|\band\b", tail, flags=re.IGNORECASE)
        products = [p.strip(" .") for p in parts if p.strip()]
        # keep Title Case-ish chunks as product candidates
        return [p for p in products if len(p.split()) <= 5]

    @staticmethod
    def extract_products_from_entities_text(entities_text: str) -> List[str]:
        """
        If you choose to parse the verbose 'metric_details.contexts' text block,
        pull out lines like 'Product:Apple Juice' from the 'Entities:' section.
        """
        products: List[str] = []
        for line in entities_text.splitlines():
            line = line.strip()
            if line.startswith("Product:"):
                products.append(line.replace("Product:", "").strip())
        return products

    @staticmethod
    def extract_relationships(relationships_text: str) -> List[Tuple[str, str, str]]:
        """
        From a 'Relationships:' text block, return tuples like (lhs, rel, rhs)
        e.g., 'Product:Carrot Juice HAS_CHARACTERISTIC Characteristic:100% pure, squeezed'
        """
        triples: List[Tuple[str, str, str]] = []
        for line in relationships_text.splitlines():
            line = line.strip()
            if not line or " " not in line:
                continue
            # split into three parts at first two spaces
            # safer: split by known relationship separators
            m = re.match(r"(.+?)\s+(HAS_\w+|BROUGHT_BY|TARGETS)\s+(.+)", line)
            if m:
                triples.append((m.group(1), m.group(2), m.group(3)))
        return triples

    @staticmethod
    def pull_entities_sections(metric_contexts: str) -> Tuple[str, str]:
        """
        From the long 'metric_details.contexts' string, slice out the
        'Entities:' and 'Relationships:' text blocks if present.
        """
        entities_block, rels_block = "", ""
        if "Entities:" in metric_contexts:
            entities_block = metric_contexts.split("Entities:")[-1].split("----")[0].strip()
        if "Relationships:" in metric_contexts:
            rels_block = metric_contexts.split("Relationships:")[-1].split("Document end")[0].strip()
        return entities_block, rels_block

    @staticmethod
    def short_summary(kg_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize what we care about for Rasa slots: message + products + specs/traits.
        """
        data = kg_json.get("data", {}) if isinstance(kg_json, dict) else {}
        message = data.get("message", "") or ""
        metric = data.get("metric_details", {}) or {}
        ctx = metric.get("contexts", "") or ""

        entities_block, rels_block = KGClient.pull_entities_sections(ctx)
        products_from_entities = KGClient.extract_products_from_entities_text(entities_block)
        products_from_msg = KGClient.extract_products_from_message(message)
        products = sorted(set(products_from_entities + products_from_msg))

        rel_triples = KGClient.extract_relationships(rels_block)
        # Build a minimal specs dict per product from relationships
        specs: Dict[str, Dict[str, List[str]]] = {}
        for lhs, rel, rhs in rel_triples:
            if lhs.startswith("Product:"):
                pname = lhs.split("Product:", 1)[-1].strip()
                bucket = specs.setdefault(pname, {})
                bucket.setdefault(rel, []).append(rhs)

        return {
            "message": message,
            "products": products,
            "specs_by_product": specs,   # e.g., {"Carrot Juice": {"HAS_BENEFIT": [...], "HAS_INGREDIENT": [...]} }
            "raw_entities": data.get("entities", {}),
            "mode": data.get("mode"),
            "sources": (data.get("info") or {}).get("sources") or [],
        }
