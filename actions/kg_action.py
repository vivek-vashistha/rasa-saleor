# actions/kg_action.py
import logging
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType
from .kg_client import KGClient

logger = logging.getLogger(__name__)

class ActionQueryKnowledgeGraph(Action):
    def name(self) -> Text:
        return "action_query_knowledge_graph"

    def __init__(self) -> None:
        self.kg = KGClient()

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        user_text = (tracker.latest_message.get("text") or "").strip()
        session_id = tracker.sender_id or "default"

        # optional per-turn overrides
        kg_mode = (tracker.get_slot("kg_mode") or "").strip()
        if kg_mode:
            self.kg.mode = kg_mode

        doc_names = tracker.get_slot("kg_document_names")
        if isinstance(doc_names, str):
            doc_names = [s.strip() for s in doc_names.split(",") if s.strip()]

        try:
            kg_json = self.kg.query(
                question=user_text,
                session_id=session_id,
                document_names=doc_names,
            )

            # Normalize key bits for downstream use
            msg = (
                kg_json.get("data", {})
                      .get("message")
                or kg_json.get("message")
                or ""
            )
            sources = (
                kg_json.get("data", {})
                       .get("info", {})
                       .get("sources")
                or []
            )

            events: List[EventType] = [
                SlotSet("kg_message", msg),
                SlotSet("kg_sources", sources),
                SlotSet("kg_raw_response", kg_json),
            ]

            if msg:
                dispatcher.utter_message(text=msg)
            else:
                dispatcher.utter_message(text="I found some info in the knowledge graph, but it didn’t include a message.")
            return events

        except Exception as e:
            logger.exception("KG action failed")
            dispatcher.utter_message(text="Sorry — I couldn’t reach our knowledge graph right now.")
            return [SlotSet("kg_error", str(e))]
