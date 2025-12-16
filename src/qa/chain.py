"""QA chain using LangChain and OpenAI."""
from langchain_openai import ChatOpenAI
try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Optional
import os
import yaml
import re
from dotenv import load_dotenv
from logging_config.logger import get_logger
from loguru import logger as loguru_logger
import json
import time

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
        
        # For spec queries, reorder chunks to prioritize important spec types
        # but keep ALL chunks - don't filter any out
        important_spec_chunks = []
        other_chunks = []
        
        for doc in retrieved_docs:
            text_lower = doc.get("text", "").lower()
            is_important = False
            
            # Check if this is an important spec chunk (Processor, RAM, Storage, Battery, Weight, Dimensions, Display)
            # Processor chunks with model names
            if (("processor" in text_lower or "cpu" in text_lower or "prozessor" in text_lower) and 
                (any(brand in text_lower for brand in ["intel", "amd", "core", "ryzen", "ultra", "i3", "i5", "i7", "i9"]) or
                 any(model in text_lower for model in ["ghz", "mhz", "cores", "kerne", "threads", "thread", "p-core", "e-core"]))):
                is_important = True
            # RAM/Memory chunks
            elif (("memory" in text_lower or "ram" in text_lower or "speicher" in text_lower or "arbeitsspeicher" in text_lower) and 
                  any(unit in text_lower for unit in ["gb", "ddr4", "ddr5", "ddr3", "sodimm"])):
                is_important = True
            # Storage chunks
            elif (("storage" in text_lower or "ssd" in text_lower or "hdd" in text_lower) and 
                  any(unit in text_lower for unit in ["gb", "tb", "m.2", "capacity"])):
                is_important = True
            # Battery chunks
            elif (("battery" in text_lower or "akku" in text_lower) and any(unit in text_lower for unit in ["w", "wh", "capacity"])):
                is_important = True
            # Weight chunks
            elif (("weight" in text_lower or "gewicht" in text_lower) and any(unit in text_lower for unit in ["kg", "lbs", "g"])):
                is_important = True
            # Dimensions chunks
            elif (("dimensions" in text_lower or "abmessungen" in text_lower) and any(unit in text_lower for unit in ["mm", "inches", "cm"])):
                is_important = True
            # Display chunks
            elif (("display" in text_lower or "screen" in text_lower) and any(unit in text_lower for unit in ["nits", "inch", "resolution", "fhd", "uhd", "4k"])):
                is_important = True
            # Graphics/GPU chunks
            elif (("graphics" in text_lower or "gpu" in text_lower or "grafik" in text_lower) and 
                  any(brand in text_lower for brand in ["intel", "amd", "nvidia", "arc", "radeon"])):
                is_important = True
            
            if is_important:
                important_spec_chunks.append(doc)
            else:
                other_chunks.append(doc)
        
        # Combine: important chunks first, then others (keep ALL chunks)
        all_chunks = important_spec_chunks + other_chunks
        
        for i, doc in enumerate(all_chunks, 1):
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
        chat_history: Optional[List[Dict]] = None,
        concise_mode: bool = False
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
        
        # Detect brand
        brand = None
        if "thinkpad" in question_lower or "think pad" in question_lower:
            brand = "ThinkPad"
        elif "zbook" in question_lower:
            brand = "HP ZBook"
        elif "ideapad" in question_lower:
            brand = "IdeaPad"
        
        # Detect model names (E14, E16, L14, P15v, etc.) - pattern: letter(s) followed by numbers
        model_pattern = r'\b([a-z]\d{1,2}|[a-z]{2}\d{1,2})\b'
        model_matches = re.findall(model_pattern, question_lower)
        
        if model_matches:
            model = max(model_matches, key=len).upper()  # Use longest match and uppercase
            
            # Detect generation number
            gen_pattern = r'\b(?:gen|generation)\s*(\d+)\b'
            gen_matches = re.findall(gen_pattern, question_lower)
            
            if gen_matches:
                gen_num = gen_matches[0]
                product_name = f"{brand} {model} Gen {gen_num}" if brand else f"{model} Gen {gen_num}"
            else:
                product_name = f"{brand} {model}" if brand else model
        
        # Handle special cases for multi-word model names
        if "zbook ultra 14" in question_lower:
            product_name = "HP ZBook Ultra 14"
        elif "zbook 8 14" in question_lower:
            product_name = "HP ZBook 8 14"
        elif "zbook 8 16" in question_lower:
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
- Suche GRÜNDLICH in ALLEN bereitgestellten Chunks nach den Informationen - auch in Chunks mit niedrigerer Ähnlichkeit!
- Durchsuche JEDEN Chunk systematisch, auch wenn er nicht direkt relevant erscheint
- Achte auf verschiedene Formulierungen und Synonyme:
  * Prozessor/CPU: "Processor", "CPU", "Prozessor", "Intel", "AMD", "Core", "Ultra", "Ryzen", "i3", "i5", "i7", "i9", "i11", "GHz", "MHz", "Cores", "Kerne", "Threads", "P-core", "E-core", "Taktfrequenz", "frequency" - Suche nach KOMPLETTEN Prozessor-Modellnamen wie "Intel Core Ultra 7 265U" oder "AMD Ryzen 5 7535U"! Wenn mehrere Prozessor-Optionen angegeben sind, nenne ALLE!
  * Storage/Speicher: "Storage", "Speicher", "SSD", "HDD", "M.2", "capacity", "Kapazität", "TB", "GB", "drive", "Laufwerk" - Achte auf "Up to X drives" oder "2x" = multipliziere die Einzelkapazität!
  * Display: "Display", "Screen", "Bildschirm", "Panel", "LCD", "IPS", "OLED", "Resolution", "Auflösung", "FHD", "UHD", "4K", "1920x1080", "2560x1440", "3840x2160"
  * Display-Helligkeit/Brightness: "Brightness", "Helligkeit", "nits", "cd/m²", "cd/m2", "luminance", "Luminanz" - Zahlen mit "nits" oder "cd/m²" sind Helligkeitsangaben!
  * Akku/Battery: "Battery", "Akku", "Batterie", "Power Adapter", "Power Supply", "W", "Wh", "Watt", "capacity", "Kapazität", "life", "Laufzeit", "hours", "Stunden", "mAh"
  * Abmessungen/Dimensions: "Dimensions", "Abmessungen", "Size", "WxDxH", "Width x Depth x Height", "mm", "inches", "cm", "Length", "Länge", "Width", "Breite", "Height", "Höhe", "Depth", "Tiefe"
  * Gewicht/Weight: "Weight", "Gewicht", "kg", "lbs", "pounds", "g", "grams"
- WICHTIG FÜR STORAGE-KAPAZITÄT: Wenn du "Up to two drives" oder "2x" siehst, multipliziere die Einzelkapazität (z.B. "2x 1TB" = 2TB maximal, NICHT 4TB!)
- Wenn Informationen in verschiedenen Chunks stehen, kombiniere sie zu einer vollständigen Antwort
- Wenn du Zahlen oder Werte siehst (z.B. "300nits", "65W", "313 x 220.3 x 10.1 mm", "1.69 kg"), verwende diese explizit
- Besonders wichtig: Wenn du "nits" oder "cd/m²" siehst, ist das eine Helligkeitsangabe - verwende diese!
- Wenn du eine Spezifikation in einem Chunk findest, auch wenn sie nicht perfekt formatiert ist, verwende sie trotzdem!
- Nur wenn du wirklich KEINE Informationen in ALLEN Chunks findest, schreibe "Nicht spezifiziert"{product_warning}{ram_instructions}

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
        
        if concise_mode:
            context_prompt += """
            
ZUSATZ-INSTRUKTION FÜR EVALUATION:
Antworte extrem kurz und präzise. Keine ganzen Sätze. Nur die angeforderten Fakten/Werte.
Beispiel Formatierung: "16 GB DDR4" statt "Der Laptop verfügt über 16 GB DDR4 Arbeitsspeicher."
"""
        
        messages.append(HumanMessage(content=context_prompt))
        
        # Get answer from LLM
        try:
            logger.info(f"Calling LLM", num_messages=len(messages), context_length=len(context))
            logger.debug(f"LLM Prompt", messages=[m.content for m in messages])
            
            response = self.llm.invoke(messages)
            
            logger.debug(f"LLM Raw Response", response=response)
            
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
        
        # Log interaction
        try:
            interaction_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": answer,
                "model": self.llm.model_name,
                "context_length": len(context) if context else 0,
                "chat_history_length": len(chat_history) if chat_history else 0
            }
            # Use bind to add the type="interaction" to extra dict
            loguru_logger.bind(type="interaction").info(json.dumps(interaction_data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

        
        result = {
            "answer": answer,
            "question": question
        }
        
        return result
    
    def answer_with_retrieved_docs(
        self,
        question: str,
        retrieved_docs: List[Dict],
        chat_history: Optional[List[Dict]] = None,
        concise_mode: bool = False
    ) -> Dict:
        """Answer question using retrieved documents and chat history."""
        # Detect if this is a general spec query (asking for all specs, not specific ones)
        question_lower = question.lower()
        is_general_spec = any(kw in question_lower for kw in ["spezifikation", "specification", "specs", "technische"]) and \
                         not any(kw in question_lower for kw in ["akku", "battery", "gewicht", "weight", "dimensions", "abmessungen", "display", "screen", "prozessor", "cpu", "ram", "memory"])
        
        # For general spec queries, ensure important spec chunks are included
        # but don't limit too aggressively - keep all retrieved docs, just reorder them
        if is_general_spec and len(retrieved_docs) > 50:
            # Separate important spec chunks from others for reordering
            important_chunks = []
            other_chunks = []
            
            for doc in retrieved_docs:
                text_lower = doc.get("text", "").lower()
                # Processor chunks with model names
                is_important = (("processor" in text_lower or "cpu" in text_lower or "prozessor" in text_lower) and 
                               (any(brand in text_lower for brand in ["intel", "amd", "core", "ryzen", "ultra", "i3", "i5", "i7", "i9"]) or
                                any(model in text_lower for model in ["ghz", "mhz", "cores", "kerne", "threads", "thread", "p-core", "e-core"]))) or \
                              (("battery" in text_lower or "akku" in text_lower) and any(unit in text_lower for unit in ["w", "wh", "capacity"])) or \
                              (("weight" in text_lower or "gewicht" in text_lower) and any(unit in text_lower for unit in ["kg", "lbs", "g"])) or \
                              (("dimensions" in text_lower or "abmessungen" in text_lower) and any(unit in text_lower for unit in ["mm", "inches", "cm"])) or \
                              (("display" in text_lower or "screen" in text_lower) and any(unit in text_lower for unit in ["nits", "inch", "resolution"])) or \
                              (("memory" in text_lower or "ram" in text_lower or "speicher" in text_lower) and any(unit in text_lower for unit in ["gb", "ddr4", "ddr5"])) or \
                              (("storage" in text_lower or "ssd" in text_lower) and any(unit in text_lower for unit in ["gb", "tb", "m.2"]))
                
                if is_important:
                    important_chunks.append(doc)
                else:
                    other_chunks.append(doc)
            
            # Reorder: important chunks first, then others (but keep ALL chunks, don't limit)
            retrieved_docs = important_chunks + other_chunks
        
        context = self.format_context(retrieved_docs)
        result = self.answer(question, context, return_sources=True, chat_history=chat_history, concise_mode=concise_mode)
        
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

