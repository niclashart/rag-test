"""Retrieval module for document search."""
from typing import List, Dict, Optional
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
import yaml
import os
import re
from logging_config.logger import get_logger

logger = get_logger(__name__)

# Cache for document filenames to avoid repeated DB queries
_document_filename_cache = {}


class Retriever:
    """Retriever for document search using vector similarity."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        config_path: str = None
    ):
        """Initialize retriever."""
        self.vector_store = vector_store
        self.embedder = embedder
        
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "./config/settings.yaml")
        
        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                retrieval_config = config.get("retrieval", {})
        else:
            retrieval_config = {}
        
        self.top_k = retrieval_config.get("top_k", 5)
        # Use very low threshold (0.0) to get all results - reranking will handle relevance
        self.similarity_threshold = retrieval_config.get("similarity_threshold", 0.0)
        self.use_reranking = retrieval_config.get("use_reranking", True)
        self.rerank_top_k = retrieval_config.get("rerank_top_k", 3)
        
        # Log the actual threshold being used
        logger.info(f"Retriever initialized with similarity_threshold={self.similarity_threshold}, top_k={self.top_k}")
    
    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        Returns list of dicts with: id, text, metadata, distance
        """
        if n_results is None:
            n_results = self.top_k
        
        # For specification queries, expand the query with technical keywords
        # Detect spec queries by keywords OR by question patterns asking for specific values
        spec_keywords = ["spezifikation", "specification", "specs", "technische", "hardware"]
        spec_question_patterns = [
            "wieviel", "wie viel", "welche", "was ist", "welcher", "welches",
            "ram", "memory", "speicher", "prozessor", "cpu", "grafik", "gpu",
            "display", "bildschirm", "akku", "battery", "gewicht", "weight",
            "abmessungen", "dimensions", "anschlüsse", "ports"
        ]
        query_lower = query.lower()
        is_spec_query = (
            any(keyword in query_lower for keyword in spec_keywords) or
            any(pattern in query_lower for pattern in spec_question_patterns)
        )
        
        # Extract product/model name from query to improve specificity
        product_keywords = []
        
        # Detect product brand
        if "thinkpad" in query_lower or "think pad" in query_lower:
            product_keywords.append("thinkpad")
        elif "zbook" in query_lower:
            product_keywords.append("zbook")
        elif "ideapad" in query_lower:
            product_keywords.append("ideapad")
        
        # Detect model names (E14, E16, L14, P15v, etc.) - pattern: letter(s) followed by numbers
        model_pattern = r'\b([a-z]\d{1,2}|[a-z]{2}\d{1,2})\b'
        model_matches = re.findall(model_pattern, query_lower)
        for model_match in model_matches:
            if model_match not in product_keywords:
                product_keywords.append(model_match)
        
        # Detect generation numbers (Gen X, Generation X, GenX) - pattern: gen/generation followed by number
        gen_pattern = r'\b(?:gen|generation)\s*(\d+)\b'
        gen_matches = re.findall(gen_pattern, query_lower)
        for gen_num in gen_matches:
            product_keywords.append(f"gen {gen_num}")
            product_keywords.append(f"generation {gen_num}")
        
        if is_spec_query:
            # Expand query with technical terms while emphasizing the specific product name
            # This helps find technical chunks while maintaining relevance to the specific product
            product_emphasis = " ".join(product_keywords) * 2 if product_keywords else ""  # Emphasize product name
            
            # Detect specific spec type to add targeted keywords
            query_lower_for_spec = query_lower
            additional_keywords = ""
            
            # Display brightness specific - CHECK THIS FIRST before RAM to avoid misclassification
            if any(term in query_lower_for_spec for term in ["helligkeit", "brightness", "nits", "luminance", "luminanz"]):
                additional_keywords = "display brightness Helligkeit nits cd/m2 cd/m² luminance luminanz screen display specifications 300 nits typical brightness"
            # RAM/Memory specific
            elif any(term in query_lower_for_spec for term in ["ram", "memory", "speicher", "arbeitsspeicher"]):
                additional_keywords = "RAM memory DDR4 DDR5 DDR3 GB gigabytes 16GB 32GB 64GB 8GB memory specifications"
            # Weight/Dimensions specific - add strong cross-language keywords
            elif any(term in query_lower_for_spec for term in ["gewicht", "weight", "schwer", "leicht", "kg", "lbs", "gramm"]):
                additional_keywords = "weight Weight Gewicht kg lbs pounds kilogram starting at mechanical dimensions size mass specifications"
            elif any(term in query_lower_for_spec for term in ["abmessung", "dimension", "größe", "maße", "breite", "höhe", "tiefe"]):
                additional_keywords = "dimensions Dimensions Abmessungen WxDxH width height depth mm inches mechanical size specifications"
            # Screen-to-Body Ratio specific - CHECK THIS BEFORE general display
            elif any(term in query_lower_for_spec for term in ["screen-to-body", "screen to body", "screen-to-body ratio", "screen to body ratio", "bezel", "display bezel", "screen bezel", "ratio"]):
                # Emphasize "Screen-to-Body Ratio" strongly and add percentage variations
                # Emphasize "Screen-to-Body Ratio" strongly and add percentage variations
                # Also add display-related keywords to find display specification chunks
                additional_keywords = "Screen-to-Body Ratio Screen to Body Ratio screen-to-body ratio screen to body ratio bezel display bezel screen bezel ratio percentage % 85% 85.5% 86% 87% 88% 88.5% 89% 90% 91% 92% 93% 94% 95% display Display screen Screen panel Panel specifications specifications table Table WUXGA FHD resolution brightness nits"
            # Display specific (not brightness)
            elif any(term in query_lower_for_spec for term in ["display", "bildschirm", "screen", "monitor", "auflösung", "resolution"]):
                additional_keywords = "display Display screen panel IPS LCD OLED FHD UHD resolution inch inches nits brightness specifications"
            # Battery/Akku specific
            elif any(term in query_lower_for_spec for term in ["akku", "battery", "batterie", "laufzeit", "wh", "kapazität"]):
                additional_keywords = "battery Battery Akku Wh capacity power cells life runtime specifications"
            else:
                # For general spec questions, add comprehensive technical keywords
                # Emphasize processor keywords strongly
                additional_keywords = "processor CPU Prozessor cores Kerne threads Threads frequency Taktfrequenz cache Intel AMD Core Ultra Ryzen i3 i5 i7 i9 GHz MHz memory RAM DDR4 DDR5 storage SSD HDD graphics GPU display screen resolution brightness nits cd/m2 luminance Helligkeit battery capacity power adapter dimensions weight size ports connectivity USB Thunderbolt HDMI"
            
            # Add generation keywords if any generation is mentioned
            generation_keywords = ""
            gen_pattern = r'\b(?:gen|generation)\s*(\d+)\b'
            gen_matches = re.findall(gen_pattern, query_lower_for_spec)
            if gen_matches:
                gen_num = gen_matches[0]  # Use first generation found
                generation_keywords = f"Gen {gen_num} Generation {gen_num} {gen_num}th generation"
            
            # Add PERFORMANCE section keywords to find technical specification chunks
            # These keywords help find chunks that contain the PERFORMANCE section with all specs
            # Emphasize PERFORMANCE section strongly for spec queries
            # Also add specific keywords for Display, Battery, Dimensions/Weight, and Processor to ensure these are found
            performance_keywords = "PERFORMANCE PERFORMANCE section PERFORMANCE specifications technical specifications processor CPU Prozessor Intel AMD Core Ultra Ryzen i3 i5 i7 i9 i11 cores Kerne threads Threads frequency Taktfrequenz GHz MHz cache memory RAM DDR4 DDR5 16GB 32GB 64GB storage SSD HDD graphics GPU display screen resolution brightness nits cd/m2 luminance Helligkeit panel IPS LCD OLED battery akku batterie power adapter W Wh capacity life dimensions abmessungen weight gewicht size width height depth mm inches kg lbs ports connectivity"
            
            expanded_query = f"{query} {product_emphasis} {additional_keywords} {generation_keywords} {performance_keywords}"
            spec_type = "RAM/memory" if additional_keywords else "general specs"
            logger.info(f"Expanded specification query (product: {product_keywords}, spec_type: {spec_type}): {expanded_query[:200]}...")
            query_embedding = self.embedder.embed_text(expanded_query)
        else:
            query_embedding = self.embedder.embed_text(query)
        
        # Query vector store - get more results for spec queries
        # For spec queries, get significantly more candidates to ensure technical chunks are found
        # Some technical chunks (like RAM specs, Display, Battery, Dimensions) might have lower similarity but are still relevant
        # Multiply by 8 instead of 6 to get even more candidates for spec queries
        query_n_results = n_results * 8 if is_spec_query else n_results
        
        logger.info(f"Querying vector store", query=query, n_results=query_n_results, is_spec_query=is_spec_query)
        
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=query_n_results,
            where=filter_metadata
        )
        
        logger.debug(f"Raw results from vector store", num_results=len(results.get('ids', [[]])[0]))
        
        # Format results and filter by product/model if specified
        retrieved_docs = []
        if results.get("ids") and len(results["ids"][0]) > 0:
            logger.info(f"ChromaDB returned {len(results['ids'][0])} results")
            
            # Extract product/model name and generation from query for filtering
            query_lower = query.lower()
            target_product = None
            target_gen = None
            
            # Detect model names (E14, E16, L14, P15v, etc.) - pattern: letter(s) followed by numbers
            model_pattern = r'\b([a-z]\d{1,2}|[a-z]{2}\d{1,2})\b'
            model_matches = re.findall(model_pattern, query_lower)
            
            if model_matches:
                # Use the first model found, but prioritize longer matches (e.g., "p15v" over "p15")
                target_product = max(model_matches, key=len)
                
                # Check for generation number
                gen_pattern = r'\b(?:gen|generation)\s*(\d+)\b'
                gen_matches = re.findall(gen_pattern, query_lower)
                if gen_matches:
                    target_gen = int(gen_matches[0])
            
            # Handle special cases for multi-word model names
            if "zbook ultra 14" in query_lower:
                target_product = "zbook ultra 14"
            elif "zbook 8 14" in query_lower:
                target_product = "zbook 8 14"
            elif "zbook 8 16" in query_lower:
                target_product = "zbook 8 16"
            
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else None
                
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                similarity = 1 - distance if distance is not None else None
                
                logger.debug(f"Result {i}: doc_id={doc_id}, distance={distance}, similarity={similarity}, threshold={self.similarity_threshold}")
                
                # Allow all results through - the threshold is mainly for logging
                # Some embedding models produce very different distance scales
                if similarity is not None and similarity >= -100.0:  # Very permissive threshold
                    doc_text = results["documents"][0][i] if results.get("documents") else ""
                    doc_metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    # For spec queries, boost similarity for chunks with technical keywords
                    text_lower = doc_text.lower()
                    similarity_boost = 0.0
                    if is_spec_query:
                        # Boost chunks that contain PERFORMANCE section or technical keywords
                        # Check for PERFORMANCE section first (most important)
                        if "performance" in text_lower and len(doc_text) > 200:  # Must be substantial chunk
                            similarity_boost = 0.25  # Very strong boost for PERFORMANCE section
                        # Processor chunks - check for processor keywords AND model names/numbers
                        # Also check for GPU/graphics tables that contain processor specifications
                        elif ((("processor" in text_lower or "cpu" in text_lower or "prozessor" in text_lower) and 
                              (any(brand in text_lower for brand in ["intel", "amd", "core", "ryzen", "ultra", "i3", "i5", "i7", "i9"]) or
                               any(model in text_lower for model in ["ghz", "mhz", "cores", "kerne", "threads", "thread", "p-core", "e-core"]))) or
                              # GPU tables often contain processor information
                              (("gpu" in text_lower or "graphics" in text_lower or "grafik" in text_lower) and 
                               any(proc_name in text_lower for proc_name in ["u300e", "i3-1315u", "core 3 100u", "core 5 120u", "core 5 220u", "core 7 150u", "core 7 250u", "core ultra 5", "core ultra 7", "processor", "intel processor"]))) and len(doc_text) > 200:
                            similarity_boost = 0.24  # Very strong boost for processor chunks with model info or GPU tables with processors
                        elif any(keyword in text_lower for keyword in ["processor", "cpu", "memory", "ram", "storage", "graphics", "gpu"]) and len(doc_text) > 200:
                            similarity_boost = 0.20  # Strong boost for technical keywords in substantial chunks
                        # Screen-to-Body Ratio chunks - check FIRST before general display chunks
                        # Boost if "Screen-to-Body Ratio" is explicitly mentioned OR if there's a percentage near display info
                        elif ("screen-to-body" in text_lower or "screen to body" in text_lower):
                            # If explicitly mentioned, boost strongly regardless of percentage format
                            similarity_boost = 0.45  # Very strong boost for explicit screen-to-body ratio mentions
                        elif (("bezel" in text_lower or "display bezel" in text_lower or "screen bezel" in text_lower) and
                              (any(ratio_term in text_lower for ratio_term in ["ratio", "%"]) or
                               any(pct in text_lower for pct in ["85%", "85.5%", "87%", "88%", "88.5%", "89%", "90%", "91%", "92%", "93%", "94%", "95%"]))):
                            similarity_boost = 0.40  # Very strong boost for bezel/ratio chunks
                        elif (("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower) and
                              any(pct in text_lower for pct in ["85%", "85.5%", "86%", "87%", "88%", "88.5%", "89%", "90%", "91%", "92%", "93%", "94%", "95%"])):
                            # Display chunks with percentage - likely screen-to-body ratio
                            similarity_boost = 0.38  # Strong boost for display chunks with percentage
                        # Display chunks - check for display keywords AND measurements/units
                        # Prioritize chunks with brightness/nits information even more
                        elif ("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower) and (
                            any(unit in text_lower for unit in ["nits", "cd/m2", "cd/m²", "brightness", "helligkeit", "luminance", "luminanz"]) or
                            ("300" in text_lower and ("nits" in text_lower or "brightness" in text_lower or "helligkeit" in text_lower))
                        ):
                            similarity_boost = 0.35  # Very strong boost for display chunks with brightness/nits
                        elif ("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower) and (
                            any(unit in text_lower for unit in ["resolution", "auflösung", "inch", "inches", "\"", "fhd", "uhd", "4k", "1920", "2560", "3840"]) or
                            any(size in text_lower for size in ["14", "15", "16"])  # Screen sizes
                        ):
                            similarity_boost = 0.30  # Strong boost for display chunks with other measurements
                        # Battery chunks - check for battery keywords AND capacity/units
                        elif ("battery" in text_lower or "akku" in text_lower or "batterie" in text_lower or "power adapter" in text_lower or "power supply" in text_lower) and (
                            any(unit in text_lower for unit in ["w", "wh", "watt", "capacity", "kapazität", "life", "laufzeit", "mah", "mah", "hours", "stunden"])
                        ):
                            similarity_boost = 0.28  # Very strong boost for battery chunks with capacity info
                        # Dimensions/Weight chunks - check for keywords AND measurements
                        elif ("dimensions" in text_lower or "abmessungen" in text_lower or "size" in text_lower or "weight" in text_lower or "gewicht" in text_lower) and (
                            any(unit in text_lower for unit in ["mm", "inches", "kg", "lbs", "pounds", "g", "x", "×", "cm"]) or
                            any(dim in text_lower for dim in ["width", "height", "depth", "breite", "höhe", "tiefe", "length", "länge"])
                        ):
                            similarity_boost = 0.26  # Strong boost for dimensions/weight chunks with measurements
                        elif ("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower):
                            similarity_boost = 0.19  # Strong boost for display chunks without measurements
                        elif "dimensions" in text_lower or ("weight" in text_lower and any(unit in text_lower for unit in ["kg", "lbs", "mm", "inches"])):
                            similarity_boost = 0.18  # Strong boost for dimensions/weight chunks
                        elif "battery" in text_lower or "akku" in text_lower or ("power adapter" in text_lower and any(unit in text_lower for unit in ["w", "wh"])):
                            similarity_boost = 0.18  # Strong boost for battery/power adapter chunks
                        elif "performance" in text_lower:
                            similarity_boost = 0.15  # Moderate boost for PERFORMANCE in smaller chunks
                        elif any(keyword in text_lower for keyword in ["processor", "cpu", "memory", "ram", "storage", "graphics", "gpu", "display", "battery", "dimensions", "weight"]):
                            similarity_boost = 0.10  # Moderate boost for technical keywords
                        
                        # Apply boost (cap at 1.0)
                        similarity = min(1.0, similarity + similarity_boost)
                    
                    # Detect if this is a processor query - we need to be more lenient with filtering
                    is_processor_query_here = any(term in query_lower for term in [
                        "prozessor", "prozessoren", "processor", "processors", "cpu", "cpus",
                        "welche prozessor", "welche processor", "prozessor-konfiguration",
                        "prozessoroptionen", "prozessor-optionen"
                    ])
                    
                    # Detect if this is a screen-to-body ratio query - we need to be more lenient with filtering
                    is_screen_to_body_query_here = any(term in query_lower for term in [
                        "screen-to-body", "screen to body", "screen-to-body ratio", "screen to body ratio",
                        "bezel", "display bezel", "screen bezel"
                    ])
                    
                    # Check if this chunk contains processor-related content
                    is_processor_chunk = (
                        ("processor" in text_lower or "prozessor" in text_lower or "cpu" in text_lower) and
                        (any(brand in text_lower for brand in ["intel", "amd", "core", "ryzen", "ultra", "i3", "i5", "i7", "i9"]) or
                         any(model in text_lower for model in ["ghz", "mhz", "cores", "kerne", "threads", "thread", "p-core", "e-core"]))
                    ) or (
                        ("|" in doc_text or "---" in doc_text or "table" in text_lower) and
                        any(keyword in text_lower for keyword in ["processor", "cpu", "core", "ultra", "intel", "amd"])
                    )
                    
                    # Check if this chunk contains screen-to-body ratio or display-related content
                    # For screen-to-body queries, accept chunks with:
                    # 1. Explicit "Screen-to-Body Ratio" mention, OR
                    # 2. Display/Screen keywords + percentage, OR
                    # 3. Just Display/Screen keywords (for general display chunks that might contain the info)
                    is_display_chunk = False
                    if is_screen_to_body_query_here:
                        # For screen-to-body queries, be more lenient
                        is_display_chunk = (
                            ("screen-to-body" in text_lower or "screen to body" in text_lower) or  # Explicit mention
                            (("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower) and
                             ("%" in doc_text or
                              any(pct in text_lower for pct in ["85%", "85.5%", "86%", "87%", "88%", "88.5%", "89%", "90%", "91%", "92%", "93%", "94%", "95%"]))) or  # Display with percentage
                            ("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower)  # General display chunks
                        )
                    else:
                        # For other queries, only check for explicit display info
                        is_display_chunk = (
                            ("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower) and
                            (("screen-to-body" in text_lower or "screen to body" in text_lower) or
                             "%" in doc_text or
                             any(pct in text_lower for pct in ["85%", "85.5%", "86%", "87%", "88%", "88.5%", "89%", "90%", "91%", "92%", "93%", "94%", "95%"]))
                        )
                    
                    # Filter by product name if target product is specified
                    if target_product:
                        # Check if chunk contains the target product name
                        # Normalize target_product for comparison (lowercase, handle variations)
                        target_product_normalized = target_product.lower()
                        has_product_in_text = target_product_normalized in text_lower
                        
                        # Also check for common variations (e.g., "thinkpad e14", "e14 gen 6")
                        if "thinkpad" in query_lower or "think pad" in query_lower:
                            has_product_in_text = has_product_in_text or f"thinkpad {target_product_normalized}" in text_lower
                        
                        # CRITICAL: For queries with generation specified, filter intelligently
                        # This prevents mixing specs from different models/generations while allowing relevant chunks
                        # For processor queries, be more lenient - accept chunks from correct document even if they don't explicitly mention model/gen
                        if target_gen is not None:
                            # Check if generation is mentioned in chunk
                            gen_in_text = (
                                f"gen {target_gen}" in text_lower or 
                                f"gen{target_gen}" in text_lower or
                                f"generation {target_gen}" in text_lower or
                                f"{target_gen}th generation" in text_lower or
                                f"{target_gen}th gen" in text_lower
                            )
                            
                            # Check document filename/metadata for model/generation hints
                            # First try to get filename from metadata, if not available, get from database cache
                            doc_source = doc_metadata.get("source", "").lower()
                            
                            # If source is not in metadata, try to get filename from database using document_id
                            if not doc_source and doc_metadata.get("document_id"):
                                document_id = doc_metadata.get("document_id")
                                # Check cache first
                                if document_id not in _document_filename_cache:
                                    try:
                                        # Import here to avoid circular dependencies
                                        from database.database import SessionLocal
                                        from database.models import Document
                                        db = SessionLocal()
                                        try:
                                            doc = db.query(Document).filter(Document.id == document_id).first()
                                            if doc:
                                                _document_filename_cache[document_id] = doc.filename.lower()
                                            else:
                                                _document_filename_cache[document_id] = ""
                                        finally:
                                            db.close()
                                    except Exception as e:
                                        logger.debug(f"Failed to fetch filename for document_id {document_id}: {e}")
                                        _document_filename_cache[document_id] = ""
                                
                                doc_source = _document_filename_cache.get(document_id, "").lower()
                            
                            filename_has_model = target_product_normalized in doc_source if doc_source else False
                            filename_has_gen = (
                                (f"gen_{target_gen}" in doc_source or 
                                 f"gen{target_gen}" in doc_source or
                                 f"generation_{target_gen}" in doc_source) if doc_source else False
                            )
                            
                            # Extract all model patterns from text to check for conflicts
                            other_models = []
                            model_pattern_in_text = r'\b([a-z]\d{1,2}|[a-z]{2}\d{1,2})\b'
                            text_model_matches = re.findall(model_pattern_in_text, text_lower)
                            for text_model in text_model_matches:
                                if text_model.lower() != target_product_normalized:
                                    # Check if this other model has a generation mentioned
                                    other_gen_pattern = rf'\b(?:gen|generation)\s*(\d+)\b'
                                    other_gen_matches = re.findall(other_gen_pattern, text_lower)
                                    if other_gen_matches:
                                        other_models.append((text_model, other_gen_matches[0]))
                            
                            # CRITICAL: Exclude if another model with different generation is explicitly mentioned
                            has_conflicting_model = any(
                                model != target_product_normalized and gen != str(target_gen) 
                                for model, gen in other_models
                            )
                            
                            if has_conflicting_model:
                                # Chunk mentions another model with different generation - exclude it
                                logger.debug(f"Filtered out result {i} - conflicting model/generation in chunk (target: {target_product} Gen {target_gen}, found: {other_models})")
                                continue
                            
                            # BALANCED APPROACH: Accept chunk if:
                            # 1. Model AND generation are in text, OR
                            # 2. Model is in text AND filename suggests correct model/gen, OR
                            # 3. Model is in text AND no conflicting models found (for technical chunks from correct doc), OR
                            # 4. Filename suggests correct model/gen AND no conflicting models (for table chunks that may not repeat model/gen in every row)
                            # 5. For processor queries: Filename suggests correct model/gen AND chunk contains processor info (even if model/gen not in text)
                            # 6. For screen-to-body ratio queries: Filename suggests correct model/gen AND chunk contains display/screen-to-body info (even if model/gen not in text)
                            # This allows technical chunks (like PERFORMANCE sections) and table chunks that may not explicitly mention generation
                            if is_processor_query_here and is_processor_chunk:
                                # For processor queries, be more lenient - accept if filename matches and chunk contains processor info
                                product_in_chunk = (
                                    (has_product_in_text and gen_in_text) or  # Explicit model + gen in text
                                    (has_product_in_text and filename_has_model and filename_has_gen) or  # Model in text + filename suggests correct doc
                                    (has_product_in_text and len(other_models) == 0) or  # Model in text, no other models mentioned
                                    (filename_has_model and filename_has_gen and len(other_models) == 0) or  # Filename suggests correct doc, no conflicting models
                                    (filename_has_model and filename_has_gen and is_processor_chunk and len(other_models) == 0)  # Filename matches + processor chunk + no conflicts
                                )
                            elif is_screen_to_body_query_here:
                                # For screen-to-body ratio queries, be VERY lenient - accept display chunks from correct document
                                # Accept if:
                                # 1. Filename matches (most important - ensures correct document)
                                # 2. Display chunk from correct document (even without explicit model/gen in text)
                                # 3. Explicit model/gen in text
                                # This is necessary because display specs often don't repeat model/gen in every chunk
                                product_in_chunk = (
                                    (filename_has_model and filename_has_gen) or  # Filename matches - accept all chunks from correct document
                                    (has_product_in_text and gen_in_text) or  # Explicit model + gen in text
                                    (has_product_in_text and filename_has_model and filename_has_gen) or  # Model in text + filename suggests correct doc
                                    (has_product_in_text and len(other_models) == 0) or  # Model in text, no other models mentioned
                                    (filename_has_model and filename_has_gen and is_display_chunk) or  # Filename matches + display chunk (even with conflicts)
                                    (filename_has_model and filename_has_gen and len(other_models) == 0)  # Filename suggests correct doc, no conflicting models
                                )
                            else:
                                product_in_chunk = (
                                    (has_product_in_text and gen_in_text) or  # Explicit model + gen in text
                                    (has_product_in_text and filename_has_model and filename_has_gen) or  # Model in text + filename suggests correct doc
                                    (has_product_in_text and len(other_models) == 0) or  # Model in text, no other models mentioned
                                    (filename_has_model and filename_has_gen and len(other_models) == 0)  # Filename suggests correct doc, no conflicting models (for table chunks)
                                )
                            
                            # Log filtering decisions - use INFO level for important chunks
                            if not product_in_chunk:
                                chunk_id_short = doc_id[:8] if doc_id else "?"
                                # Check if this might be the graphics table chunk
                                is_graphics_table = ("graphics" in text_lower or "gpu" in text_lower) and ("table" in text_lower or "|" in doc_text or "---" in doc_text)
                                log_level = logger.info if (is_graphics_table or doc_id == "bd9d0fc1-98f4-4ebe-a47c-eed250205951") else logger.debug
                                # Show filename from cache if available, otherwise from metadata
                                displayed_filename = doc_source[:50] if doc_source else (doc_metadata.get('source', '?')[:50] if doc_metadata.get('source') else '?')
                                log_level(f"Chunk {chunk_id_short}... filtered out - has_product_in_text: {has_product_in_text}, gen_in_text: {gen_in_text}, filename_match: {filename_has_model and filename_has_gen}, other_models: {len(other_models)}, target: {target_product} Gen {target_gen}, filename: {displayed_filename}")
                        else:
                            # Without generation, require explicit product mention
                            product_in_chunk = has_product_in_text
                        
                        # If product filtering is active but chunk doesn't match, skip it
                        if not product_in_chunk:
                            logger.debug(f"Filtered out result {i} - product/generation mismatch (target: {target_product}{f' Gen {target_gen}' if target_gen else ''}, similarity: {similarity})")
                            continue
                    
                    retrieved_docs.append({
                        "id": doc_id,
                        "text": doc_text,
                        "metadata": doc_metadata,
                        "distance": distance,
                        "similarity": similarity
                    })
                else:
                    logger.debug(f"Filtered out result {i} due to similarity threshold")
            
            # Sort by similarity (highest first) if we applied boosts
            # For spec queries, prioritize important spec types (Display, Battery, Dimensions) even if similarity is slightly lower
            if is_spec_query:
                # Check if this is a screen-to-body ratio query (outside sort_key to avoid repeated checks)
                is_screen_to_body_query = any(term in query_lower for term in [
                    "screen-to-body", "screen to body", "screen-to-body ratio", "screen to body ratio",
                    "bezel", "display bezel", "screen bezel"
                ])
                
                def sort_key(doc):
                    similarity = doc.get("similarity", 0)
                    text_lower = doc.get("text", "").lower()
                    text = doc.get("text", "")
                    
                    # Priority boost for important spec types
                    priority = 0
                    
                    # HIGHEST PRIORITY: Explicit Screen-to-Body Ratio mentions (for screen-to-body queries)
                    if is_screen_to_body_query and ("screen-to-body" in text_lower or "screen to body" in text_lower):
                        priority = 1000  # Very high priority - should be at the top
                    elif is_screen_to_body_query and ("display" in text_lower or "screen" in text_lower) and "%" in text:
                        # Display chunks with percentage for screen-to-body queries
                        priority = 500  # High priority
                    elif "performance" in text_lower and len(doc.get("text", "")) > 200:
                        priority = 1000  # Highest priority
                    # Processor chunks with model names get very high priority
                    elif (("processor" in text_lower or "cpu" in text_lower or "prozessor" in text_lower) and 
                          (any(brand in text_lower for brand in ["intel", "amd", "core", "ryzen", "ultra", "i3", "i5", "i7", "i9"]) or
                           any(model in text_lower for model in ["ghz", "mhz", "cores", "kerne", "threads", "thread", "p-core", "e-core"]))):
                        priority = 950  # Very high priority for processor with model info
                    elif ("battery" in text_lower or "akku" in text_lower) and any(unit in text_lower for unit in ["w", "wh", "capacity"]):
                        priority = 900  # Very high priority for battery
                    elif ("weight" in text_lower or "gewicht" in text_lower) and any(unit in text_lower for unit in ["kg", "lbs", "g"]):
                        priority = 850  # Very high priority for weight
                    elif ("dimensions" in text_lower or "abmessungen" in text_lower) and any(unit in text_lower for unit in ["mm", "inches", "cm"]):
                        priority = 800  # High priority for dimensions
                    # Prioritize display chunks with brightness/nits even more
                    elif ("display" in text_lower or "screen" in text_lower) and (
                        any(unit in text_lower for unit in ["nits", "brightness", "helligkeit", "luminance"]) or
                        ("300" in text_lower and ("nits" in text_lower or "brightness" in text_lower))
                    ):
                        priority = 850  # Very high priority for display with brightness
                    elif ("display" in text_lower or "screen" in text_lower) and any(unit in text_lower for unit in ["inch", "resolution"]):
                        priority = 750  # High priority for display
                    elif any(kw in text_lower for kw in ["processor", "cpu", "memory", "ram", "storage"]):
                        priority = 700  # Medium-high priority for core specs
                    
                    # Return combined score (priority + similarity)
                    return priority + similarity
                
                retrieved_docs.sort(key=sort_key, reverse=True)
        else:
            logger.warning(f"No results returned from ChromaDB")
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query (after threshold and product filtering)")
        return retrieved_docs
    
    def retrieve_with_reranking(
        self,
        user_id: int,
        query: str,
        reranker,
        n_results: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve documents and rerank them.
        Returns reranked list of documents.
        """
        # First retrieve more documents for reranking (get more candidates)
        target_k = n_results or self.rerank_top_k
        
        # For specification questions, retrieve even more candidates
        # Use same detection logic as in retrieve() method
        spec_keywords = ["spezifikation", "specification", "specs", "technische", "hardware"]
        spec_question_patterns = [
            "wieviel", "wie viel", "welche", "was ist", "welcher", "welches",
            "ram", "memory", "speicher", "prozessor", "cpu", "grafik", "gpu",
            "display", "bildschirm", "akku", "battery", "gewicht", "weight",
            "abmessungen", "dimensions", "anschlüsse", "ports"
        ]
        query_lower = query.lower()
        is_spec_query = (
            any(keyword in query_lower for keyword in spec_keywords) or
            any(pattern in query_lower for pattern in spec_question_patterns)
        )
        
        # Detect processor/CPU questions specifically - these often need multiple chunks
        # because processor tables can span multiple chunks
        is_processor_query = any(term in query_lower for term in [
            "prozessor", "prozessoren", "processor", "processors", "cpu", "cpus",
            "welche prozessor", "welche processor", "prozessor-konfiguration",
            "prozessoroptionen", "prozessor-optionen"
        ])
        
        # Detect screen-to-body ratio questions - these may need more chunks to find the ratio information
        is_screen_to_body_query = any(term in query_lower for term in [
            "screen-to-body", "screen to body", "screen-to-body ratio", "screen to body ratio",
            "bezel", "display bezel", "screen bezel"
        ])
        
        if is_processor_query:
            # For processor questions, return significantly more chunks (30-50)
            # because processor tables are often split across multiple chunks
            target_k = max(target_k, 50)  # At least 50 chunks for processor queries
            initial_k = max(target_k * 8, 80)  # Get 8x more candidates
            logger.info(f"Processor query detected, retrieving {initial_k} candidates, returning top {target_k}")
        elif is_screen_to_body_query:
            # For screen-to-body ratio questions, get more chunks to find ratio information
            target_k = max(target_k, 40)  # At least 40 chunks for screen-to-body queries
            initial_k = max(target_k * 8, 60)  # Get 8x more candidates
            logger.info(f"Screen-to-body ratio query detected, retrieving {initial_k} candidates, returning top {target_k}")
        elif is_spec_query:
            # For spec questions, get even more candidates to ensure we find technical chunks
            initial_k = max(target_k * 8, 40)  # Get 8x more for spec questions to ensure technical chunks are found
            logger.info(f"Specification query detected, retrieving {initial_k} candidates")
        else:
            initial_k = max(target_k * 3, 10)  # Get 3x more for reranking, minimum 10
        
        retrieved = self.retrieve(query, n_results=initial_k)
        
        logger.info(f"Retrieved {len(retrieved)} documents for reranking (target: {target_k})")
        
        if not self.use_reranking or not reranker:
            return retrieved[:target_k]
        
        # Rerank
        if len(retrieved) > 0:
            query_texts = [doc["text"] for doc in retrieved]
            try:
                reranked_indices = reranker.rerank(query, query_texts, top_k=target_k)
                reranked_docs = [retrieved[i] for i in reranked_indices]
                logger.info(f"Reranked {len(retrieved)} documents to top {len(reranked_docs)}")
                return reranked_docs
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, returning original results")
                return retrieved[:target_k]
        
        return retrieved


