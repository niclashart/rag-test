"""QA chain using LangChain and OpenAI."""
from langchain_openai import ChatOpenAI
try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Optional
import os
import yaml
from dotenv import load_dotenv
from logging_config.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class QAChain:
    """Question-answering chain with context."""
    
    def __init__(self, config_path: str = None):
        """Initialize QA chain."""
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "./config/settings.yaml")
        
        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                qa_config = config.get("qa", {})
        else:
            qa_config = {}
        
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Use gpt-4o-mini as default (faster, cheaper)
        temperature = qa_config.get("temperature", 0.7)
        max_tokens = qa_config.get("max_tokens", 2000)  # Increased to allow for longer answers
        system_prompt = qa_config.get("system_prompt", 
            "Du bist ein hilfreicher Assistent, der Fragen basierend auf den bereitgestellten Dokumenten beantwortet.")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.system_prompt = system_prompt
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents as context."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            chunk_id = doc.get("id", "")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source_info = metadata.get("source", "")
            
            context_parts.append(
                f"[Quelle {i} - {chunk_id}]:\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def answer(
        self,
        question: str,
        context: str,
        return_sources: bool = True,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Answer a question based on context and chat history.
        Returns dict with answer and optionally sources.
        """
        # Build messages with chat history
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-10:]:  # Keep last 10 messages for context
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # Detect if question is about specifications
        spec_keywords = [
            "spezifikation", "specification", "specs", 
            "technische daten", "technischen daten", "technischen details",
            "hardware", "ausstattung", "konfiguration",
            "prozessor", "cpu", "prozessoroptionen",
            "ram", "arbeitsspeicher", "memory",
            "speicher", "storage", "ssd", "hdd",
            "grafik", "gpu", "grafikkarte", "graphics",
            "display", "bildschirm", "screen",
            "akku", "battery", "batterie",
            "abmessungen", "gewicht", "weight", "dimensions",
            "anschlüsse", "ports", "connectivity"
        ]
        # Also detect question patterns asking for specific values
        spec_question_patterns = [
            "wieviel", "wie viel", "welche", "was ist", "welcher", "welches"
        ]
        question_lower = question.lower()
        is_spec_question = (
            any(keyword in question_lower for keyword in spec_keywords) or
            any(pattern in question_lower for pattern in spec_question_patterns)
        )
        
        # Extract product/model name from question for context
        question_lower = question.lower()
        product_name = None
        if "e16" in question_lower:
            if "gen 3" in question_lower or "gen3" in question_lower:
                product_name = "ThinkPad E16 Gen 3"
            else:
                product_name = "ThinkPad E16"
        elif "e14" in question_lower:
            if "gen 7" in question_lower or "generation 7" in question_lower:
                product_name = "ThinkPad E14 Gen 7"
            elif "gen 6" in question_lower or "generation 6" in question_lower:
                product_name = "ThinkPad E14 Gen 6"
            else:
                product_name = "ThinkPad E14"
        elif "p15v" in question_lower or ("p15" in question_lower and "v" in question_lower):
            product_name = "ThinkPad P15v"
        elif "zbook" in question_lower:
            if "ultra 14" in question_lower:
                product_name = "HP ZBook Ultra 14"
            elif "8 14" in question_lower:
                product_name = "HP ZBook 8 14"
            elif "8 16" in question_lower:
                product_name = "HP ZBook 8 16"
        
        # Add current context and question
        if is_spec_question:
            if product_name:
                product_warning = f"""

KRITISCH WICHTIG - MODELL-SPEZIFIZITÄT:
Die Frage bezieht sich AUSSCHLIESSLICH auf das {product_name}.
- Verwende NUR Informationen, die explizit für das {product_name} genannt werden
- Wenn ein Chunk Informationen über andere Modelle enthält (z.B. E14, E16 Gen 2, P15v, andere ThinkPad-Modelle, andere ZBook-Modelle), IGNORIERE diese komplett
- Wenn ein Chunk mehrere Modelle erwähnt, verwende NUR die Informationen für das {product_name}
- Wenn du dir nicht sicher bist, ob eine Information zum {product_name} gehört, erwähne sie NICHT
- Vermische KEINE Spezifikationen von verschiedenen Modellen oder Generationen
- Wenn eine Information nicht eindeutig dem {product_name} zugeordnet werden kann, lasse sie weg
- Prüfe IMMER, ob eine Zahl oder Spezifikation wirklich zum {product_name} gehört, bevor du sie verwendest"""
            else:
                product_warning = "\n\nWICHTIG: Verwende nur Informationen, die eindeutig dem in der Frage genannten Modell zugeordnet werden können. Vermische keine Informationen von verschiedenen Modellen."
            
            # Detect if question is specifically about RAM/memory
            is_ram_question = any(term in question_lower for term in ["ram", "memory", "speicher", "arbeitsspeicher"])
            
            ram_instructions = ""
            if is_ram_question:
                ram_instructions = """

WICHTIG FÜR RAM/SPEICHER-FRAGEN:
- Suche in ALLEN bereitgestellten Dokumenten nach RAM/Memory-Spezifikationen
- Achte auf verschiedene Formulierungen: "Memory", "RAM", "Total Memory", "Up to XGB", "DDR4", "DDR5"
- Wenn mehrere RAM-Optionen angegeben sind (z.B. "4GB, 8GB, 12GB oder 16GB"), nenne alle Optionen
- Wenn "Up to XGB" angegeben ist, ist das die maximale RAM-Kapazität
- Gib auch den RAM-Typ an (z.B. DDR4, DDR5) und die Geschwindigkeit (z.B. 2666MHz, 3200MHz) wenn verfügbar
- Wenn die RAM-Informationen in verschiedenen Chunks stehen, kombiniere sie zu einer vollständigen Antwort"""
            
            context_prompt = f"""Du bist ein präziser Dokumenten-Assistent. Beantworte die Frage NUR mit Informationen, die EXPLIZIT in den bereitgestellten Dokumenten stehen.

STRENGE REGELN - KEINE ERFINDUNGEN:
1. Verwende NUR Text, der wortwörtlich oder sinngemäß in den Dokumenten steht
2. Erfinde KEINE Zahlen, Werte oder Spezifikationen, auch nicht wenn du sie aus deinem Training kennst
3. Wenn eine Information NICHT in den Dokumenten steht, schreibe explizit: "Nicht in den Dokumenten angegeben" oder "Nicht spezifiziert"
4. Verwende KEINE allgemeinen Kenntnisse oder typischen Werte für ähnliche Produkte
5. Wenn du dir nicht 100% sicher bist, dass eine Information in den Dokumenten steht, erwähne sie NICHT
6. Zitiere die Quelle (z.B. "Quelle 1") wenn möglich

WICHTIG: Nenne zuerst die wichtigsten technischen Spezifikationen (Prozessor, RAM, Storage, Grafik, Display, Akku, Abmessungen/Gewicht, Anschlüsse). 
Erwähne weniger wichtige Details (Webcam, Mikrofone, LEDs, etc.) erst am Ende oder gar nicht.

WICHTIG FÜR ALLE SPEZIFIKATIONEN:
- Suche GRÜNDLICH in ALLEN bereitgestellten Chunks nach den Informationen
- Achte auf verschiedene Formulierungen und Synonyme:
  * Storage/Speicher: "Storage", "Speicher", "SSD", "HDD", "M.2", "capacity", "Kapazität", "TB", "GB" - Achte auf "Up to X drives" oder "2x" = multipliziere die Einzelkapazität!
  * Display-Helligkeit/Brightness: "Brightness", "Helligkeit", "nits", "cd/m²", "cd/m2", "luminance", "Luminanz" - Zahlen mit "nits" oder "cd/m²" sind Helligkeitsangaben!
  * Akku/Battery: "Battery", "Akku", "Batterie", "Power Adapter", "W", "Wh", "capacity", "life"
  * Abmessungen/Dimensions: "Dimensions", "Abmessungen", "Size", "WxDxH", "Width x Depth x Height", "mm", "inches"
  * Gewicht/Weight: "Weight", "Gewicht", "kg", "lbs", "pounds"
- WICHTIG FÜR STORAGE-KAPAZITÄT: Wenn du "Up to two drives" oder "2x" siehst, multipliziere die Einzelkapazität (z.B. "2x 1TB" = 2TB maximal, NICHT 4TB!)
- Wenn Informationen in verschiedenen Chunks stehen, kombiniere sie zu einer vollständigen Antwort
- Wenn du Zahlen oder Werte siehst (z.B. "300nits", "65W", "313 x 220.3 x 10.1 mm", "1.69 kg"), verwende diese explizit
- Besonders wichtig: Wenn du "nits" oder "cd/m²" siehst, ist das eine Helligkeitsangabe - verwende diese!
- Nur wenn du wirklich KEINE Informationen findest, schreibe "Nicht spezifiziert"{product_warning}{ram_instructions}

Dokumente:
{context}

Frage: {question}

Antwort (NUR dokumentierte Informationen verwenden - wenn etwas nicht dokumentiert ist, explizit "Nicht spezifiziert" angeben):"""
        else:
            context_prompt = f"""Basierend auf den folgenden Dokumenten, beantworte die Frage.

Dokumente:
{context}

Frage: {question}

Antwort:"""
        
        messages.append(HumanMessage(content=context_prompt))
        
        # Get answer from LLM
        try:
            logger.info(f"Calling LLM with {len(messages)} messages, context length: {len(context)}")
            response = self.llm.invoke(messages)
            
            # Handle different response types
            if hasattr(response, 'content'):
                answer = response.content
            elif hasattr(response, 'text'):
                answer = response.text
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)
            
            logger.info(f"LLM response type: {type(response)}, answer length: {len(answer) if answer else 0}")
            
            if not answer or answer.strip() == "":
                logger.warning(f"LLM returned empty answer. Response object: {response}")
                # Try to get more info
                if hasattr(response, '__dict__'):
                    logger.warning(f"Response attributes: {response.__dict__}")
                answer = "Entschuldigung, ich konnte keine Antwort generieren. Bitte versuchen Sie es mit einer anderen Formulierung."
            
            logger.info(f"Generated answer for question: {question[:50]}... (length: {len(answer)})")
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            answer = f"Fehler bei der Generierung der Antwort: {str(e)}"
        
        result = {
            "answer": answer,
            "question": question
        }
        
        return result
    
    def answer_with_retrieved_docs(
        self,
        question: str,
        retrieved_docs: List[Dict],
        chat_history: Optional[List[Dict]] = None
    ) -> Dict:
        """Answer question using retrieved documents and chat history."""
        context = self.format_context(retrieved_docs)
        result = self.answer(question, context, return_sources=True, chat_history=chat_history)
        
        # Add source information
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "chunk_id": doc.get("id"),
                "document_id": doc.get("metadata", {}).get("document_id"),
                "page_number": doc.get("metadata", {}).get("page_number"),
                "similarity": doc.get("similarity")
            })
        result["sources"] = sources
        
        return result

