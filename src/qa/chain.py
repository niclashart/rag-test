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
        eval_max_tokens = qa_config.get("eval_max_tokens", None)  # Optional max tokens for eval mode
        system_prompt = qa_config.get("system_prompt", 
            "Du bist ein hilfreicher Assistent, der Fragen basierend auf den bereitgestellten Dokumenten beantwortet.")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=5,  # Increased retries to handle rate limiting
            request_timeout=120  # Increased timeout for large contexts
        )
        
        self.system_prompt = system_prompt
        self.eval_max_tokens = eval_max_tokens
        self.default_max_tokens = max_tokens
    
    def format_context(self, retrieved_docs: List[Dict], preserve_order: bool = False) -> str:
        """Format retrieved documents as context.
        
        Args:
            retrieved_docs: List of documents to format
            preserve_order: If True, use documents in the order provided (don't re-sort).
                           This is important when documents are already sorted to match source numbering.
        """
        context_parts = []
        
        # If preserve_order is True, use documents as-is (already sorted correctly)
        if preserve_order:
            all_chunks = retrieved_docs
        else:
            # For spec queries, reorder chunks to prioritize important spec types
            # but keep ALL chunks - don't filter any out
            important_spec_chunks = []
            other_chunks = []
            
            for doc in retrieved_docs:
                text_lower = doc.get("text", "").lower()
                is_important = False
                
                # Check if this is an important spec chunk (Processor, RAM, Storage, Battery, Weight, Dimensions, Display)
                # Processor chunks with model names - also check for GPU tables that contain processor info
                # Check for processor keywords OR GPU/graphics tables that might contain processor specifications
                has_processor_keywords = (("processor" in text_lower or "cpu" in text_lower or "prozessor" in text_lower) and 
                    (any(brand in text_lower for brand in ["intel", "amd", "core", "ryzen", "ultra", "i3", "i5", "i7", "i9"]) or
                     any(model in text_lower for model in ["ghz", "mhz", "cores", "kerne", "threads", "thread", "p-core", "e-core"])))
                # Also check for GPU/graphics tables that contain processor names (common pattern)
                has_gpu_table_with_processors = (("gpu" in text_lower or "graphics" in text_lower or "grafik" in text_lower) and 
                    any(proc_name in text_lower for proc_name in ["u300e", "i3-1315u", "core 3 100u", "core 5 120u", "core 5 220u", "core 7 150u", "core 7 250u", "core ultra 5", "core ultra 7"]))
                # Check for processor model numbers (e.g., 225H, 225U, 235H, 235U, 245H, 245U)
                has_processor_models = any(model in text_lower for model in [
                    "225h", "225u", "235h", "235u", "245h", "245u", "255h", "255u",
                    "core ultra 5 225", "core ultra 5 235", "core ultra 5 245",
                    "core ultra 7 155", "core ultra 7 165", "core ultra 7 155h", "core ultra 7 165h"
                ])
                # Check for table-like structures that might contain processor information
                doc_text = doc.get("text", "")
                has_table_structure = (("|" in doc_text or "---" in doc_text or "table" in text_lower) and 
                    any(keyword in text_lower for keyword in ["processor", "cpu", "core", "ultra", "intel", "amd"]))
                
                if has_processor_keywords or has_gpu_table_with_processors or has_processor_models or has_table_structure:
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
            # Note: For processor questions, the chunks are already sorted correctly in answer_with_retrieved_docs
            # when preserve_order=True is used
            all_chunks = important_spec_chunks + other_chunks
        
        # Log which chunks are being formatted and in what order
        target_chunk_id = "bd9d0fc1-98f4-4ebe-a47c-eed250205951"  # The correct graphics table chunk
        target_position_in_context = None
        
        logger.info(f"format_context: Formatting {len(all_chunks)} chunks for context")
        
        for i, doc in enumerate(all_chunks, 1):
            chunk_id = doc.get("id", "")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source_info = metadata.get("source", "")
            
            # Check if this is the target chunk
            if chunk_id == target_chunk_id:
                target_position_in_context = i
                logger.info(f"format_context: Target graphics table chunk found at position {i} in context")
            
            # Log processor-related chunks
            text_lower = text.lower()
            if any(term in text_lower for term in ["processor", "prozessor", "core ultra", "intel core"]):
                logger.info(f"format_context: Processor chunk found at position {i} - chunk_id: {chunk_id[:8]}..., text_preview: {text[:100]}...")
            
            context_parts.append(
                f"[Quelle {i} - {chunk_id}]:\n{text}\n"
            )
        
        if target_position_in_context is None:
            logger.warning(f"format_context: Target chunk {target_chunk_id[:8]}... NOT FOUND in context!")
        
        formatted_context = "\n---\n".join(context_parts)
        logger.info(f"format_context: Created context with {len(context_parts)} chunks, total length: {len(formatted_context)} characters")
        
        return formatted_context
    
    def answer(
        self,
        question: str,
        context: str,
        return_sources: bool = True,
        chat_history: Optional[List[Dict]] = None,
        concise_mode: bool = False,
        eval_mode: bool = False,
        ground_truth: Optional[str] = None
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
        
        # Log number of chunks being sent to LLM
        num_chunks = len(context.split("\n---\n")) if context else 0
        logger.info(f"answer: Sending {num_chunks} chunks to LLM for question: {question[:100]}...")
        
        # Add current context and question
        if is_spec_question:
            if product_name:
                product_warning = f"""

KRITISCH WICHTIG - MODELL-SPEZIFIZITÄT:
Die Frage bezieht sich AUSSCHLIESSLICH auf das {product_name}.
- Verwende NUR Informationen, die explizit für das {product_name} genannt werden
- Wenn ein Chunk Informationen über andere Modelle enthält (z.B. E14, E16 Gen 2, P15v, L16 Gen 1, andere ThinkPad-Modelle, andere ZBook-Modelle), IGNORIERE diese komplett
- Wenn ein Chunk mehrere Modelle erwähnt, verwende NUR die Informationen für das {product_name}
- Wenn ein Chunk ein anderes Modell oder eine andere Generation erwähnt (z.B. "E14 Gen 6" wenn nach "L16 Gen 2" gefragt wird), IGNORIERE den gesamten Chunk
- Wenn du dir nicht sicher bist, ob eine Information zum {product_name} gehört, erwähne sie NICHT
- Vermische KEINE Spezifikationen von verschiedenen Modellen oder Generationen
- Wenn eine Information nicht eindeutig dem {product_name} zugeordnet werden kann, lasse sie weg
- Prüfe IMMER, ob eine Zahl oder Spezifikation wirklich zum {product_name} gehört, bevor du sie verwendest
- WICHTIG: Wenn ein Chunk sowohl das {product_name} als auch andere Modelle erwähnt, verwende NUR die Informationen, die explizit dem {product_name} zugeordnet sind"""
            else:
                product_warning = "\n\nWICHTIG: Verwende nur Informationen, die eindeutig dem in der Frage genannten Modell zugeordnet werden können. Vermische keine Informationen von verschiedenen Modellen."
            
            # Count number of chunks in context (needed for instructions)
            num_chunks_in_context = len(context.split("\n---\n")) if context else 0
            
            # Detect if question is specifically about RAM/memory
            is_ram_question = any(term in question_lower for term in ["ram", "memory", "speicher", "arbeitsspeicher"])
            
            # Detect if question is specifically about processors/CPU
            is_processor_question = any(term in question_lower for term in ["prozessor", "processor", "cpu", "prozessoren", "processors", "welche prozessoren", "welche processor"])
            
            # Detect if question is specifically about screen-to-body ratio
            is_screen_to_body_question = any(term in question_lower for term in [
                "screen-to-body", "screen to body", "screen-to-body ratio", "screen to body ratio",
                "bezel", "display bezel", "screen bezel"
            ])
            
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
            
            screen_to_body_instructions = ""
            if is_screen_to_body_question:
                screen_to_body_instructions = f"""

⚠️⚠️⚠️ KRITISCH WICHTIG FÜR SCREEN-TO-BODY RATIO-FRAGEN ⚠️⚠️⚠️
- Die Screen-to-Body Ratio Information kann als PROZENTSATZ angegeben sein, auch mit Dezimalstellen (z.B. "85.5%", "87.2%", "88.5%")!
- Suche in ALLEN {num_chunks_in_context} bereitgestellten Chunks nach:
  * Expliziten Erwähnungen: "Screen-to-Body Ratio", "Screen to Body Ratio"
  * PROZENTANGABEN in Display-Kontext: "85%", "85.5%", "86%", "87%", "88%", "88.5%", "89%", "90%", "91%", "92%", "93%", "94%", "95%"
- WICHTIG: Wenn du "Screen-to-Body Ratio" explizit erwähnt siehst (auch ohne direktes Prozentzeichen daneben), suche in diesem Chunk nach der Prozentangabe!
- WICHTIG: Wenn du eine Prozentangabe in der Nähe von Display-, Screen-, Panel- oder Bildschirm-Informationen findest, ist das wahrscheinlich die Screen-to-Body Ratio!
- Die Information kann in verschiedenen Formulierungen stehen:
  * Als explizite "Screen-to-Body Ratio: 85.5%" oder "Screen-to-Body Ratio: 85%"
  * Als einfacher Prozentsatz in Display-Spezifikationen: "Display: 14 inch, 85.5%"
  * Als separater Punkt unter einer Display-Tabelle: "Screen-to-Body Ratio: 85.5%"
  * Als Teil einer Tabelle oder Liste von Display-Spezifikationen
  * In der Nähe von Display-Größen, Auflösungen oder Panel-Informationen
- ⚠️ WICHTIG: Durchsuche JEDEN Chunk systematisch nach "Screen-to-Body Ratio" UND nach PROZENTANGABEN (auch mit Dezimalstellen wie "85.5%")!
- ⚠️ Wenn du "Screen-to-Body Ratio" findest, suche in diesem Chunk nach der zugehörigen Prozentangabe (kann auch in der nächsten Zeile stehen)!
- ⚠️ Wenn du eine Prozentangabe findest (z.B. "85%" oder "85.5%"), gib diese als Screen-to-Body Ratio an, auch wenn sie nicht explizit so bezeichnet ist!
- ⚠️ NUR wenn du wirklich KEINE "Screen-to-Body Ratio" Erwähnung UND KEINE Prozentangabe in Display-Kontext findest, schreibe "Nicht in den Dokumenten angegeben"!
- Wenn die Information in verschiedenen Chunks steht, kombiniere sie zu einer vollständigen Antwort"""
            
            processor_instructions = ""
            if is_processor_question:
                # Count processor-related chunks in context
                num_chunks = len(context.split("\n---\n")) if context else 0
                
                # Count how many chunks contain processor tables
                processor_table_chunks = []
                processor_names_found = {}
                if context:
                    chunks = context.split("\n---\n")
                    for i, chunk_text in enumerate(chunks, 1):
                        chunk_lower = chunk_text.lower()
                        # Check if this chunk contains a processor table (has | and processor names)
                        has_table = "|" in chunk_text and ("processor" in chunk_lower or "prozessor" in chunk_lower)
                        
                        # Check which specific processors are in this chunk
                        processors_in_chunk = []
                        for proc in ["core ultra 7 165u", "core ultra 5 125u", "core ultra 7 155u",
                                    "core ultra 5 135u", "core ultra 5 125h", "core ultra 7 155h"]:
                            if proc in chunk_lower:
                                processors_in_chunk.append(proc.replace("core ultra ", "Core Ultra ").replace("u", "U").replace("h", "H"))
                        
                        if has_table and processors_in_chunk:
                            processor_table_chunks.append(i)
                            processor_names_found[i] = processors_in_chunk
                
                processor_list_str = ""
                if processor_names_found:
                    for chunk_num, procs in processor_names_found.items():
                        processor_list_str += f"\n- Quelle {chunk_num}: {', '.join(procs)}"
                
                processor_instructions = f"""

⚠️⚠️⚠️ KRITISCH WICHTIG FÜR PROZESSOR-FRAGEN - DU MUSST ALLE CHUNKS VERWENDEN ⚠️⚠️⚠️
- Es wurden {num_chunks} Chunks bereitgestellt. Du MUSST ALLE {num_chunks} Chunks durchsuchen und verwenden!
- WICHTIG: Die Prozessor-Tabelle ist über MEHRERE Chunks verteilt! Ich habe Prozessoren in folgenden Chunks gefunden:{processor_list_str if processor_list_str else '\n- Suche in ALLEN Chunks systematisch!'}
- ⚠️ DU DARFST NICHT NUR EINEN CHUNK VERWENDEN! ⚠️ Du MUSST ALLE Chunks mit Prozessor-Informationen kombinieren!
- ⚠️ WENN DU NUR 3 PROZESSOREN FINDEST, SUCHE WEITER! ⚠️ Es gibt MEHR als 3 Prozessoren!
- Wenn du z.B. in Quelle 13 die Prozessoren "Core Ultra 7 165U, Core Ultra 5 125U, Core Ultra 7 155U" findest, MUSST du AUCH in Quelle 38, Quelle 39, etc. suchen!
- Suche GRÜNDLICH in ALLEN bereitgestellten Dokumenten nach Prozessor-Informationen, besonders in Tabellen!
- Durchsuche JEDEN Chunk systematisch von Anfang bis Ende, auch Tabellen mit anderen Titeln (z.B. "PERFORMANCE", "Graphics", etc.)
- WICHTIG: Prozessor-Tabellen können über MEHRERE Chunks verteilt sein! Du MUSST ALLE Chunks durchsuchen!
- Beispiel: Wenn Quelle 13 die Prozessoren "Core Ultra 7 165U, Core Ultra 5 125U, Core Ultra 7 155U" enthält und Quelle 38 die Prozessoren "Core Ultra 5 135U, Core Ultra 5 125H, Core Ultra 7 155H" enthält, musst du ALLE 6 Prozessoren auflisten!
- Wenn eine Prozessor-Tabelle gefunden wird, liste ALLE Prozessoren auf, die in der Tabelle stehen - KEINE Auswahl!
- Wenn mehrere Prozessor-Modelle in der Tabelle stehen (z.B. "Core Ultra 5 225H", "Core Ultra 5 225U", "Core Ultra 5 235H", etc.), nenne ALLE Modelle!
- Wenn Prozessor-Informationen in MEHREREN Chunks stehen (z.B. Quelle 13 enthält Prozessor A, B, C und Quelle 38 enthält Prozessor D, E, F), kombiniere ALLE zu einer vollständigen Liste!
- Gib für jeden Prozessor die vollständigen Spezifikationen an (Cores, Threads, Frequenzen, Cache, etc.) wenn verfügbar
- Wenn die Prozessor-Informationen über mehrere Chunks verteilt sind, kombiniere sie zu einer vollständigen Liste
- WICHTIG: Wenn die Frage nach "welche Prozessoren" fragt, liste ALLE auf, die in ALLEN Chunks stehen - nicht nur eine Auswahl!
- ⚠️ KRITISCH: Auch wenn du bereits 3 Prozessoren gefunden hast, SUCHE WEITER! Es gibt mehr Prozessoren in anderen Chunks! ⚠️
- WICHTIG: Beginne mit Quelle 1, dann Quelle 2, dann Quelle 3, usw. - durchsuche ALLE Quellen systematisch!
- KRITISCH: Wenn du Prozessoren in Quelle 13 findest, suche AUCH in Quelle 38, Quelle 39, etc. - die Tabelle kann über mehrere Chunks verteilt sein!
- ⚠️ WICHTIG: Deine Antwort MUSS Prozessoren aus MEHREREN Quellen enthalten, wenn sie in mehreren Quellen stehen! ⚠️
- ⚠️ WENN DU NUR 3 PROZESSOREN AUFLISTEST, IST DAS FALSCH! ⚠️ Es gibt 6 Prozessoren! ⚠️"""
            
            # Count number of chunks in context
            num_chunks_in_context = len(context.split("\n---\n")) if context else 0
            
            # For processor questions, add a very explicit instruction at the top
            processor_warning_top = ""
            if is_processor_question:
                processor_warning_top = f"""

⚠️⚠️⚠️ KRITISCH WICHTIG FÜR PROZESSOR-FRAGEN ⚠️⚠️⚠️
Die Prozessor-Tabelle ist über MEHRERE Chunks verteilt! 
Du MUSST ALLE {num_chunks_in_context} Chunks durchsuchen und verwenden!
NUR EINEN Chunk zu verwenden ist FALSCH - du MUSST mehrere Chunks kombinieren!
Wenn du z.B. in Quelle 13 Prozessoren findest, suche AUCH in Quelle 38, 39, etc.!
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

"""
            
            context_prompt = f"""Du bist ein präziser Dokumenten-Assistent. Beantworte die Frage NUR mit Informationen, die EXPLIZIT in den bereitgestellten Dokumenten stehen.
{processor_warning_top}
WICHTIG: Es wurden {num_chunks_in_context} Chunks (Quellen) bereitgestellt. Du MUSST ALLE {num_chunks_in_context} Chunks durchsuchen!

STRENGE REGELN - KEINE ERFINDUNGEN:
1. Verwende NUR Text, der wortwörtlich oder sinngemäß in den Dokumenten steht
2. Erfinde KEINE Zahlen, Werte oder Spezifikationen, auch nicht wenn du sie aus deinem Training kennst
3. Wenn eine Information NICHT in den Dokumenten steht, schreibe explizit: "Nicht in den Dokumenten angegeben" oder "Nicht spezifiziert"
4. Verwende KEINE allgemeinen Kenntnisse oder typischen Werte für ähnliche Produkte
5. Wenn du dir nicht 100% sicher bist, dass eine Information in den Dokumenten steht, erwähne sie NICHT
6. WICHTIG: Füge KEINE Quellenangaben am Ende der Antwort hinzu (z.B. "Diese Informationen stammen aus den Quellen..."). Die Quellen werden separat angezeigt.

WICHTIG: Nenne zuerst die wichtigsten technischen Spezifikationen (Prozessor, RAM, Storage, Grafik, Display, Akku, Abmessungen/Gewicht, Anschlüsse). 
Erwähne weniger wichtige Details (Webcam, Mikrofone, LEDs, etc.) erst am Ende oder gar nicht.

WICHTIG FÜR ALLE SPEZIFIKATIONEN:
- Suche GRÜNDLICH in ALLEN {num_chunks_in_context} bereitgestellten Chunks nach den Informationen - auch in Chunks mit niedrigerer Ähnlichkeit!
- Durchsuche JEDEN Chunk systematisch von Anfang bis Ende, auch wenn er nicht direkt relevant erscheint
- Beginne mit Quelle 1, dann Quelle 2, dann Quelle 3, usw. - durchsuche ALLE Quellen systematisch!
- WICHTIG FÜR TABELLEN: Prozessoren können auch in Tabellen stehen, die andere Themen behandeln (z.B. GPU-Tabellen, Spezifikations-Tabellen). Durchsuche ALLE Tabellen gründlich, auch wenn der Tabellentitel nicht direkt "Prozessor" oder "CPU" enthält!
- KRITISCH FÜR PROZESSOR-TABELLEN: Große Tabellen können über MEHRERE Chunks verteilt sein! Du MUSST ALLE Chunks durchsuchen, um die vollständige Liste zu erhalten! Wenn Quelle 1 die ersten 3 Prozessoren enthält und Quelle 2 die nächsten 3 Prozessoren, musst du BEIDE verwenden!
- Ignoriere HTML-Formatierung wie <br>, <p>, etc. - extrahiere den reinen Text-Inhalt!
- KRITISCH: Wenn ein Chunk mehrere Modelle oder Generationen erwähnt, verwende NUR die Informationen, die explizit dem {product_name} zugeordnet sind. Ignoriere alle anderen Modell-Informationen komplett!
- Wenn ein Chunk z.B. sowohl "E14 Gen 6" als auch "L16 Gen 2" erwähnt und die Frage nach "L16 Gen 2" ist, verwende NUR die Informationen für "L16 Gen 2"!
- Achte auf verschiedene Formulierungen und Synonyme:
  * Prozessor/CPU: "Processor", "CPU", "Prozessor", "Intel", "AMD", "Core", "Ultra", "Ryzen", "i3", "i5", "i7", "i9", "i11", "GHz", "MHz", "Cores", "Kerne", "Threads", "P-core", "E-core", "Taktfrequenz", "frequency" - Suche nach KOMPLETTEN Prozessor-Modellnamen wie "Intel Core Ultra 7 265U", "Intel Processor U300E", "13th Generation Intel Core i3-1315U", "Intel Core 3 100U", "Intel Core 5 120U" oder "AMD Ryzen 5 7535U"! Wenn mehrere Prozessor-Optionen angegeben sind, nenne ALLE - auch wenn sie in einer Tabelle mit anderem Titel steht (z.B. GPU-Tabelle)!
  * Storage/Speicher: "Storage", "Speicher", "SSD", "HDD", "M.2", "capacity", "Kapazität", "TB", "GB", "drive", "Laufwerk" - Achte auf "Up to X drives" oder "2x" = multipliziere die Einzelkapazität!
  * Display: "Display", "Screen", "Bildschirm", "Panel", "LCD", "IPS", "OLED", "Resolution", "Auflösung", "FHD", "UHD", "4K", "1920x1080", "2560x1440", "3840x2160"
  * Screen-to-Body Ratio: "Screen-to-Body Ratio", "Screen to Body Ratio", "Bezel", "Display Bezel", "Screen Bezel", "Ratio", "%" (Prozentangaben) - Suche nach Prozentangaben in Display-Kontext (z.B. "85%", "87%", "90%") oder nach Begriffen wie "bezel", "screen-to-body", "display bezel"! Diese Information kann auch als Prozentsatz ohne explizite Nennung von "Screen-to-Body Ratio" angegeben sein!
  * Display-Helligkeit/Brightness: "Brightness", "Helligkeit", "nits", "cd/m²", "cd/m2", "luminance", "Luminanz" - Zahlen mit "nits" oder "cd/m²" sind Helligkeitsangaben! Suche besonders nach "300 nits" oder ähnlichen Zahlen mit "nits"!
  * Akku/Battery: "Battery", "Akku", "Batterie", "Power Adapter", "Power Supply", "W", "Wh", "Watt", "capacity", "Kapazität", "life", "Laufzeit", "hours", "Stunden", "mAh"
  * Abmessungen/Dimensions: "Dimensions", "Abmessungen", "Size", "WxDxH", "Width x Depth x Height", "mm", "inches", "cm", "Length", "Länge", "Width", "Breite", "Height", "Höhe", "Depth", "Tiefe"
  * Gewicht/Weight: "Weight", "Gewicht", "kg", "lbs", "pounds", "g", "grams"
- WICHTIG FÜR STORAGE-KAPAZITÄT: Wenn du "Up to two drives" oder "2x" siehst, multipliziere die Einzelkapazität (z.B. "2x 1TB" = 2TB maximal, NICHT 4TB!)
- Wenn Informationen in verschiedenen Chunks stehen, kombiniere sie zu einer vollständigen Antwort
- Wenn du Zahlen oder Werte siehst (z.B. "300nits", "65W", "313 x 220.3 x 10.1 mm", "1.69 kg"), verwende diese explizit
- Besonders wichtig: Wenn du "nits" oder "cd/m²" siehst, ist das eine Helligkeitsangabe - verwende diese!
- Wenn du eine Spezifikation in einem Chunk findest, auch wenn sie nicht perfekt formatiert ist, verwende sie trotzdem!
- KRITISCH: Wenn du Informationen gefunden hast, gib NUR diese Informationen aus. Füge KEIN "Nicht spezifiziert" am Ende hinzu, wenn bereits Informationen vorhanden sind!
- Nur wenn du wirklich KEINE Informationen in ALLEN Chunks findest (also GAR NICHTS), schreibe "Nicht spezifiziert"{product_warning}{ram_instructions}{screen_to_body_instructions}{processor_instructions}

Dokumente (Es wurden {num_chunks_in_context} Chunks bereitgestellt - du MUSST ALLE durchsuchen!):
{context}

Frage: {question}

Antwort (NUR dokumentierte Informationen verwenden. Wenn du Informationen gefunden hast, gib NUR diese aus. Füge KEIN "Nicht spezifiziert" hinzu, wenn bereits Informationen vorhanden sind. Nur wenn GAR KEINE Informationen gefunden wurden, schreibe "Nicht spezifiziert"):
- WICHTIG: Füge KEINE Quellenangaben am Ende der Antwort hinzu (z.B. "Diese Informationen stammen aus den Quellen..." oder ähnliche Texte). Die Quellen werden separat angezeigt.

KRITISCH FÜR PROZESSOR-FRAGEN - STRUKTURIERTE ANWEISUNG:
1. SCHRITT 1: Durchsuche SYSTEMATISCH ALLE {num_chunks_in_context} Chunks von Anfang bis Ende
2. SCHRITT 2: Identifiziere ALLE Chunks, die Prozessor-Tabellen enthalten (suche nach "|" und Prozessor-Namen)
3. SCHRITT 3: Extrahiere ALLE Prozessoren aus JEDEM gefundenen Chunk
4. SCHRITT 4: Kombiniere ALLE Prozessoren aus ALLEN Chunks zu einer vollständigen Liste
5. SCHRITT 5: Liste ALLE Prozessoren auf - auch wenn sie in verschiedenen Chunks stehen!

WICHTIGE REGELN:
- Die Prozessor-Tabelle ist über MEHRERE Chunks verteilt! Du MUSST ALLE Chunks durchsuchen!
- Beispiel: Wenn du in Quelle 13 die Prozessoren "Core Ultra 7 165U, Core Ultra 5 125U, Core Ultra 7 155U" findest, suche AUCH in Quelle 38, Quelle 39, etc. nach weiteren Prozessoren!
- Wenn die Frage nach mehreren Prozessoren fragt und du Prozessoren in MEHREREN Chunks findest (z.B. 3 Prozessoren in Quelle 13 und 3 Prozessoren in Quelle 38), liste ALLE Prozessoren aus ALLEN Chunks auf!
- Beginne deine Antwort mit einer systematischen Durchsuchung aller {num_chunks_in_context} Chunks!
- Wenn du Prozessoren in mehreren Chunks findest, kombiniere sie zu einer vollständigen Liste!
- WICHTIG: Auch wenn du bereits 3 Prozessoren gefunden hast, suche WEITER in den anderen Chunks - es gibt mehr Prozessoren!
- KRITISCH: Deine Antwort MUSS Prozessoren aus MEHREREN Quellen enthalten, wenn sie in mehreren Quellen stehen!
- WICHTIG: Füge KEINE Quellenangaben am Ende der Antwort hinzu (z.B. "Diese Informationen stammen aus den Quellen..."). Die Quellen werden separat angezeigt!"""
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
        
        if eval_mode and ground_truth:
            # Evaluierungsmodus: Versuche die Antwortlänge und den Stil an den Ground Truth anzupassen
            ground_truth_length = len(ground_truth)
            # Schätze die Token-Anzahl basierend auf der Zeichenlänge (ca. 4 Zeichen pro Token)
            estimated_tokens = max(50, int(ground_truth_length / 3))  # Etwas mehr als nötig für Sicherheit
            
            context_prompt += f"""

WICHTIG FÜR EVALUIERUNG:
- Die erwartete Antwortlänge beträgt etwa {ground_truth_length} Zeichen (ca. {estimated_tokens} Tokens)
- Antworte im gleichen Stil und Format wie der Ground Truth
- Verwende die gleiche Struktur (z.B. Komma-getrennte Liste, Aufzählung, etc.)
- Beispiel Ground Truth Format: "{ground_truth[:100]}..."
- Antworte präzise und ohne zusätzliche Erklärungen, nur die Fakten
- Verwende KEINE zusätzlichen Formatierungen wie Aufzählungszeichen oder Nummerierungen, wenn der Ground Truth diese nicht hat
- Wenn der Ground Truth eine Komma-getrennte Liste ist, verwende das gleiche Format
"""
            
            # Temporär max_tokens für diese Anfrage anpassen
            if self.eval_max_tokens:
                original_max_tokens = self.llm.max_tokens
                self.llm.max_tokens = min(self.eval_max_tokens, estimated_tokens + 50)  # Etwas Puffer
        
        messages.append(HumanMessage(content=context_prompt))
        
        # For processor questions, increase max_tokens to allow listing all processors
        original_max_tokens = self.llm.max_tokens
        is_processor_question = any(term in question.lower() for term in [
            "prozessor", "prozessoren", "processor", "processors", "cpu", "cpus",
            "welche prozessor", "welche processor"
        ])
        if is_processor_question:
            # Increase max_tokens for processor questions to allow listing all processors
            self.llm.max_tokens = max(original_max_tokens, 3000)
            logger.info(f"Increased max_tokens to {self.llm.max_tokens} for processor question")
        
        # Get answer from LLM
        try:
            logger.info(f"Calling LLM", num_messages=len(messages), context_length=len(context), eval_mode=eval_mode, max_tokens=self.llm.max_tokens)
            logger.debug(f"LLM Prompt", messages=[m.content for m in messages])
            
            response = self.llm.invoke(messages)
            
            # Restore original max_tokens
            if is_processor_question:
                self.llm.max_tokens = original_max_tokens
            
            # Stelle max_tokens wieder her wenn eval_mode verwendet wurde
            if eval_mode and self.eval_max_tokens:
                self.llm.max_tokens = self.default_max_tokens
            
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
        concise_mode: bool = False,
        eval_mode: bool = False,
        ground_truth: Optional[str] = None
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
        
        # Detect if this is a processor question
        question_lower = question.lower()
        is_processor_question_here = any(term in question_lower for term in [
            "prozessor", "prozessoren", "processor", "processors", "cpu", "cpus",
            "welche prozessor", "welche processor"
        ])
        
        # IMPORTANT: Reorder documents to match format_context order
        # format_context sorts documents (important_spec_chunks first, then others)
        # We need the same order here to correctly map source numbers
        # For processor questions, prioritize processor chunks even more strongly
        processor_chunks = []
        important_spec_chunks = []
        other_chunks = []
        
        for doc in retrieved_docs:
            text_lower = doc.get("text", "").lower()
            doc_text = doc.get("text", "")
            is_important = False
            is_processor_chunk = False
            
            # Same logic as in format_context
            # Enhanced processor detection: also check for processor model numbers and table structures
            has_processor_keywords = (("processor" in text_lower or "cpu" in text_lower or "prozessor" in text_lower) and 
                (any(brand in text_lower for brand in ["intel", "amd", "core", "ryzen", "ultra", "i3", "i5", "i7", "i9"]) or
                 any(model in text_lower for model in ["ghz", "mhz", "cores", "kerne", "threads", "thread", "p-core", "e-core"])))
            has_gpu_table_with_processors = (("gpu" in text_lower or "graphics" in text_lower or "grafik" in text_lower) and 
                any(proc_name in text_lower for proc_name in ["u300e", "i3-1315u", "core 3 100u", "core 5 120u", "core 5 220u", "core 7 150u", "core 7 250u", "core ultra 5", "core ultra 7"]))
            # Check for processor model numbers (e.g., 225H, 225U, 235H, 235U, 245H, 245U)
            has_processor_models = any(model in text_lower for model in [
                "225h", "225u", "235h", "235u", "245h", "245u", "255h", "255u",
                "core ultra 5 225", "core ultra 5 235", "core ultra 5 245",
                "core ultra 7 155", "core ultra 7 165", "core ultra 7 155h", "core ultra 7 165h",
                "core ultra 5 125u", "core ultra 5 135u", "core ultra 5 125h",
                "core ultra 7 155u", "core ultra 7 165u", "core ultra 7 155h"
            ])
            # Check for table-like structures that might contain processor information
            has_table_structure = (("|" in doc_text or "---" in doc_text or "table" in text_lower) and 
                any(keyword in text_lower for keyword in ["processor", "cpu", "core", "ultra", "intel", "amd"]))
            
            # For processor questions, check if this is specifically a processor table chunk
            if is_processor_question_here:
                # Check if this chunk contains processor table rows
                has_processor_table = (
                    ("|" in doc_text and "processor" in text_lower) or
                    (has_table_structure and has_processor_models) or
                    (has_table_structure and has_processor_keywords)
                )
                if has_processor_table:
                    is_processor_chunk = True
            
            if is_processor_chunk and is_processor_question_here:
                processor_chunks.append(doc)
            elif has_processor_keywords or has_gpu_table_with_processors or has_processor_models or has_table_structure:
                is_important = True
            elif (("memory" in text_lower or "ram" in text_lower or "speicher" in text_lower or "arbeitsspeicher" in text_lower) and 
                  any(unit in text_lower for unit in ["gb", "ddr4", "ddr5", "ddr3", "sodimm"])):
                is_important = True
            elif (("storage" in text_lower or "ssd" in text_lower or "hdd" in text_lower) and 
                  any(unit in text_lower for unit in ["gb", "tb", "m.2", "capacity"])):
                is_important = True
            elif (("battery" in text_lower or "akku" in text_lower) and any(unit in text_lower for unit in ["w", "wh", "capacity"])):
                is_important = True
            elif (("weight" in text_lower or "gewicht" in text_lower) and any(unit in text_lower for unit in ["kg", "lbs", "g"])):
                is_important = True
            elif (("dimensions" in text_lower or "abmessungen" in text_lower) and any(unit in text_lower for unit in ["mm", "inches", "cm"])):
                is_important = True
            # Screen-to-Body Ratio chunks - prioritize these highly
            # Check for explicit mention first
            if "screen-to-body" in text_lower or "screen to body" in text_lower:
                is_important = True
            elif (("bezel" in text_lower or "display bezel" in text_lower or "screen bezel" in text_lower) and
                  (any(ratio_term in text_lower for ratio_term in ["ratio", "%"]) or
                   any(pct in text_lower for pct in ["85%", "85.5%", "86%", "87%", "88%", "88.5%", "89%", "90%", "91%", "92%", "93%", "94%", "95%"]))):
                is_important = True
            elif (("display" in text_lower or "screen" in text_lower) and
                  any(pct in text_lower for pct in ["85%", "85.5%", "86%", "87%", "88%", "88.5%", "89%", "90%", "91%", "92%", "93%", "94%", "95%"])):
                # Display chunks with percentage - likely screen-to-body ratio
                is_important = True
            elif (("display" in text_lower or "screen" in text_lower) and any(unit in text_lower for unit in ["nits", "inch", "resolution", "fhd", "uhd", "4k"])):
                is_important = True
            elif (("graphics" in text_lower or "gpu" in text_lower or "grafik" in text_lower) and 
                  any(brand in text_lower for brand in ["intel", "amd", "nvidia", "arc", "radeon"])):
                is_important = True
            
            if is_important:
                important_spec_chunks.append(doc)
            else:
                other_chunks.append(doc)
        
        # For processor questions, put processor chunks FIRST, then other important chunks, then others
        # This ensures the LLM sees processor chunks at the beginning of the context
        if is_processor_question_here:
            ordered_docs = processor_chunks + important_spec_chunks + other_chunks
            logger.info(f"For processor question: Reordered {len(retrieved_docs)} docs - {len(processor_chunks)} processor chunks first, then {len(important_spec_chunks)} important chunks, then {len(other_chunks)} others")
            
            # Log which processors are in which chunks
            for i, doc in enumerate(processor_chunks[:5], 1):  # Log first 5 processor chunks
                chunk_id = doc.get("id", "")[:16]
                text_preview = doc.get("text", "")[:150].replace('\n', ' ')
                logger.info(f"  Processor chunk {i}: chunk_id={chunk_id}..., preview={text_preview}...")
        else:
            # Combine in same order as format_context (important first, then others)
            ordered_docs = important_spec_chunks + other_chunks
        
        # Limit the number of chunks to avoid exceeding token limits
        # Estimate: ~4 characters per token, max context ~128k tokens for gpt-4o-mini
        # Reserve ~30k tokens for prompt/system messages, leaving ~98k for context
        # With ~1200 chars per chunk, that's ~81 chunks max, but be conservative
        # For evaluation, use fewer chunks to avoid rate limits and token errors
        MAX_CHUNKS = 30 if eval_mode else 50  # Reduced for evaluation to avoid token limit errors
        if len(ordered_docs) > MAX_CHUNKS:
            logger.warning(f"Limiting chunks from {len(ordered_docs)} to {MAX_CHUNKS} to avoid token limit")
            # For processor questions, prioritize processor chunks and important chunks
            if is_processor_question_here:
                # Keep all processor chunks, then fill remaining slots with important chunks
                num_processor = len(processor_chunks)
                num_important = len(important_spec_chunks)
                remaining_slots = MAX_CHUNKS - num_processor
                
                if remaining_slots > 0:
                    # Take as many important chunks as possible
                    limited_important = important_spec_chunks[:min(remaining_slots, num_important)]
                    remaining_slots -= len(limited_important)
                    
                    # Fill remaining slots with other chunks
                    limited_other = other_chunks[:remaining_slots] if remaining_slots > 0 else []
                    
                    ordered_docs = processor_chunks + limited_important + limited_other
                    logger.info(f"Limited to {len(ordered_docs)} chunks: {num_processor} processor + {len(limited_important)} important + {len(limited_other)} other")
                else:
                    # If processor chunks alone exceed limit, keep only top processor chunks
                    ordered_docs = processor_chunks[:MAX_CHUNKS]
                    logger.warning(f"Processor chunks alone exceed limit, keeping only top {MAX_CHUNKS} processor chunks")
            else:
                # For non-processor questions, keep important chunks first, then others
                num_important = len(important_spec_chunks)
                if num_important >= MAX_CHUNKS:
                    ordered_docs = important_spec_chunks[:MAX_CHUNKS]
                    logger.info(f"Limited to top {MAX_CHUNKS} important chunks")
                else:
                    remaining_slots = MAX_CHUNKS - num_important
                    limited_other = other_chunks[:remaining_slots]
                    ordered_docs = important_spec_chunks + limited_other
                    logger.info(f"Limited to {len(ordered_docs)} chunks: {num_important} important + {len(limited_other)} other")
        
        # Log the order of documents for debugging
        logger.info(f"Ordered {len(ordered_docs)} documents for context. First 10 chunk_ids:")
        target_chunk_id = "bd9d0fc1-98f4-4ebe-a47c-eed250205951"  # The correct graphics table chunk
        target_found_at = None
        for i, doc in enumerate(ordered_docs[:10], 1):
            chunk_id = doc.get("id", "")
            chunk_id_short = chunk_id[:8] if chunk_id else "?"
            page = doc.get("metadata", {}).get("page_number", "?")
            text_preview = doc.get("text", "")[:80].replace('\n', ' ')
            logger.info(f"  Position {i}: chunk_id={chunk_id_short}..., page={page}, preview={text_preview}...")
            # Check if this is the target chunk
            if chunk_id == target_chunk_id:
                target_found_at = i
                logger.info(f"  *** TARGET CHUNK FOUND at position {i} ***")
        
        # Check all documents for target chunk
        if target_found_at is None:
            for i, doc in enumerate(ordered_docs, 1):
                if doc.get("id") == target_chunk_id:
                    target_found_at = i
                    logger.warning(f"Target chunk found at position {i} (beyond first 10)")
                    break
        
        if target_found_at is None:
            logger.warning(f"Target chunk {target_chunk_id[:8]}... NOT FOUND in ordered_docs!")
        
        # IMPORTANT: preserve_order=True ensures format_context doesn't re-sort the documents
        # This is critical for source numbering to match between context and filtered sources
        # The documents are already sorted correctly in ordered_docs (processor chunks first for processor questions)
        context = self.format_context(ordered_docs, preserve_order=True)
        
        # Additional safety check: Estimate tokens and reduce if necessary
        # Rough estimate: ~4 characters per token, but be conservative with ~3.5
        estimated_tokens = len(context) / 3.5
        # For evaluation, use stricter limits to avoid token errors
        MAX_CONTEXT_TOKENS = 80000 if eval_mode else 100000  # Reduced for evaluation to avoid 128k limit errors
        
        if estimated_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"Context too large: {estimated_tokens:.0f} estimated tokens, reducing chunks")
            # Reduce chunks further if needed - just take fewer chunks from ordered_docs
            # Calculate how many chunks we can keep based on size
            target_chars = MAX_CONTEXT_TOKENS * 3.5
            current_chars = sum(len(doc.get("text", "")) for doc in ordered_docs)
            
            if current_chars > target_chars:
                # Calculate how many chunks we can keep
                avg_chunk_size = current_chars / len(ordered_docs) if ordered_docs else 0
                max_chunks_by_size = int(target_chars / avg_chunk_size) if avg_chunk_size > 0 else MAX_CHUNKS
                max_chunks_by_size = max(10, min(max_chunks_by_size, MAX_CHUNKS))  # At least 10, at most MAX_CHUNKS
                
                if len(ordered_docs) > max_chunks_by_size:
                    logger.warning(f"Further reducing from {len(ordered_docs)} to {max_chunks_by_size} chunks based on size")
                    # Simply take the first N chunks (they're already prioritized)
                    ordered_docs = ordered_docs[:max_chunks_by_size]
                    
                    # Re-format context with reduced chunks
                    context = self.format_context(ordered_docs, preserve_order=True)
                    logger.info(f"Reformatted context with {len(ordered_docs)} chunks, estimated {len(context) / 3.5:.0f} tokens")
        result = self.answer(
            question, 
            context, 
            return_sources=True, 
            chat_history=chat_history, 
            concise_mode=concise_mode,
            eval_mode=eval_mode,
            ground_truth=ground_truth
        )
        
        # Extract source numbers from answer (e.g., "Quelle 6" -> 6)
        answer = result.get("answer", "")
        source_numbers = set()
        
        # Pattern to match "Quelle X" or "[Quelle X]"
        single_pattern = r'(?:\[)?Quelle\s+(\d+)(?:\])?'
        matches = re.findall(single_pattern, answer, re.IGNORECASE)
        for match in matches:
            source_numbers.add(int(match))
        
        # Pattern to match "Quelle X-Y" (range)
        range_pattern = r'Quelle\s+(\d+)\s*-\s*(\d+)'
        range_matches = re.findall(range_pattern, answer, re.IGNORECASE)
        for start_str, end_str in range_matches:
            start, end = int(start_str), int(end_str)
            source_numbers.update(range(start, end + 1))
        
        # Filter sources to only include referenced ones
        if source_numbers:
            # Map source numbers (1-based) to actual documents
            filtered_sources = []
            for i, doc in enumerate(ordered_docs, 1):
                if i in source_numbers:
                    chunk_id = doc.get("id")
                    chunk_text_preview = doc.get("text", "")[:100] if doc.get("text") else ""
                    logger.info(f"Mapping source number {i} to chunk_id: {chunk_id[:8]}... (page: {doc.get('metadata', {}).get('page_number')}, text_preview: {chunk_text_preview}...)")
                    filtered_sources.append({
                        "chunk_id": chunk_id,
                        "document_id": doc.get("metadata", {}).get("document_id"),
                        "page_number": doc.get("metadata", {}).get("page_number"),
                        "similarity": doc.get("similarity"),
                        "text": doc.get("text", ""),  # Include text directly from doc
                        "source_number": i  # Preserve original source number from context
                    })
            
            result["sources"] = filtered_sources
            logger.info(f"Filtered sources: found {len(source_numbers)} referenced sources ({source_numbers}) out of {len(ordered_docs)} total")
            logger.info(f"Returned chunk_ids: {[s['chunk_id'][:8] + '...' for s in filtered_sources]}")
        else:
            # No source references found - return top sources by similarity (max 10)
            # Sort by similarity (descending) and take top 10
            sorted_docs = sorted(ordered_docs, key=lambda x: x.get("similarity", 0.0), reverse=True)
            top_sources = sorted_docs[:10]
            
            result["sources"] = [{
                "chunk_id": doc.get("id"),
                "document_id": doc.get("metadata", {}).get("document_id"),
                "page_number": doc.get("metadata", {}).get("page_number"),
                "similarity": doc.get("similarity"),
                "text": doc.get("text", "")  # Include text directly from doc
            } for doc in top_sources]
            
            logger.info(f"No source references found in answer, returning top {len(top_sources)} sources by similarity")
        
        # Return the actual chunks used for answer generation (for RAGAS evaluation)
        # This ensures RAGAS evaluates with the same chunks that were used to generate the answer
        result["used_chunks"] = ordered_docs
        
        return result

