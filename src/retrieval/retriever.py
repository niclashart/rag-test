"""Retrieval module for document search."""
from typing import List, Dict, Optional
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
import yaml
import os
import re
from logging_config.logger import get_logger

logger = get_logger(__name__)


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
        user_id: int,
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
            
            # RAM/Memory specific
            if any(term in query_lower_for_spec for term in ["ram", "memory", "speicher", "arbeitsspeicher"]):
                additional_keywords = "RAM memory DDR4 DDR5 DDR3 GB gigabytes 16GB 32GB 64GB 8GB memory specifications"
            # Display brightness specific
            elif any(term in query_lower_for_spec for term in ["helligkeit", "brightness", "nits", "luminance", "luminanz"]):
                additional_keywords = "display brightness Helligkeit nits cd/m2 cd/m² luminance luminanz screen display specifications"
            else:
                # For general spec questions, add comprehensive technical keywords
                additional_keywords = "processor CPU cores threads frequency cache memory RAM DDR4 DDR5 storage SSD HDD graphics GPU display screen resolution brightness nits cd/m2 luminance Helligkeit battery capacity power adapter dimensions weight size ports connectivity USB Thunderbolt HDMI"
            
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
            # Also add specific keywords for Display, Battery, Dimensions/Weight to ensure these are found
            performance_keywords = "PERFORMANCE PERFORMANCE section PERFORMANCE specifications technical specifications processor CPU memory RAM DDR4 DDR5 16GB 32GB 64GB storage SSD HDD graphics GPU display screen resolution brightness nits cd/m2 luminance Helligkeit panel IPS LCD OLED battery akku batterie power adapter W Wh capacity life dimensions abmessungen weight gewicht size width height depth mm inches kg lbs ports connectivity"
            
            expanded_query = f"{query} {product_emphasis} {additional_keywords} {generation_keywords} {performance_keywords}"
            spec_type = "RAM/memory" if additional_keywords else "general specs"
            logger.info(f"Expanded specification query (product: {product_keywords}, spec_type: {spec_type}): {expanded_query[:200]}...")
            query_embedding = self.embedder.embed_text(expanded_query)
        else:
            query_embedding = self.embedder.embed_text(query)
        
        # Query vector store - get more results for spec queries
        # For spec queries, get significantly more candidates to ensure technical chunks are found
        # Some technical chunks (like RAM specs) might have lower similarity but are still relevant
        query_n_results = n_results * 6 if is_spec_query else n_results
        results = self.vector_store.query(
            user_id=user_id,
            query_embeddings=[query_embedding],
            n_results=query_n_results,
            where=filter_metadata
        )
        
        # Format results and filter by product/model if specified
        retrieved_docs = []
        if results.get("ids") and len(results["ids"][0]) > 0:
            logger.info(f"ChromaDB returned {len(results['ids'][0])} results for user {user_id}")
            
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
                
                if similarity is not None and similarity >= self.similarity_threshold:
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
                        elif any(keyword in text_lower for keyword in ["processor", "cpu", "memory", "ram", "storage", "graphics", "gpu"]) and len(doc_text) > 200:
                            similarity_boost = 0.20  # Strong boost for technical keywords in substantial chunks
                        # Display chunks - check for display keywords AND measurements/units
                        elif ("display" in text_lower or "screen" in text_lower or "bildschirm" in text_lower) and (
                            any(unit in text_lower for unit in ["nits", "cd/m2", "cd/m²", "brightness", "helligkeit", "luminance", "luminanz", "resolution", "auflösung", "inch", "inches", "\""]) or
                            any(size in text_lower for size in ["14", "15", "16"])  # Screen sizes
                        ):
                            similarity_boost = 0.22  # Very strong boost for display chunks with measurements
                        # Battery chunks - check for battery keywords AND capacity/units
                        elif ("battery" in text_lower or "akku" in text_lower or "batterie" in text_lower or "power adapter" in text_lower) and (
                            any(unit in text_lower for unit in ["w", "wh", "watt", "capacity", "kapazität", "life", "laufzeit", "mah", "mah"])
                        ):
                            similarity_boost = 0.21  # Very strong boost for battery chunks with capacity info
                        # Dimensions/Weight chunks - check for keywords AND measurements
                        elif ("dimensions" in text_lower or "abmessungen" in text_lower or "size" in text_lower or "weight" in text_lower or "gewicht" in text_lower) and (
                            any(unit in text_lower for unit in ["mm", "inches", "kg", "lbs", "pounds", "g", "x", "×"]) or
                            any(dim in text_lower for dim in ["width", "height", "depth", "breite", "höhe", "tiefe"])
                        ):
                            similarity_boost = 0.20  # Strong boost for dimensions/weight chunks with measurements
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
                    
                    # Filter by product name if target product is specified
                    if target_product:
                        # Check if chunk contains the target product name OR if it's a technical chunk from the right document
                        product_in_chunk = False
                        
                        # Check if chunk contains the target product name
                        # Normalize target_product for comparison (lowercase, handle variations)
                        target_product_normalized = target_product.lower()
                        has_product_in_text = target_product_normalized in text_lower
                        
                        # Also check for common variations (e.g., "thinkpad e14", "e14 gen 6")
                        if "thinkpad" in query_lower or "think pad" in query_lower:
                            has_product_in_text = has_product_in_text or f"thinkpad {target_product_normalized}" in text_lower
                        
                        # For queries with generation specified, also accept technical chunks
                        # even if they don't explicitly mention the model name
                        # This is important because PERFORMANCE sections and spec tables often don't include the model name
                        if target_gen is not None:
                            # Check for various technical chunk types
                            has_performance = "performance" in text_lower
                            has_core_specs = any(kw in text_lower for kw in ["processor", "cpu", "memory", "ram", "storage", "graphics", "gpu"])
                            has_display = any(kw in text_lower for kw in ["display", "screen", "bildschirm"])
                            has_battery = (("battery" in text_lower or "akku" in text_lower) and 
                                         any(unit in text_lower for unit in ["w", "wh", "capacity", "kapazität", "mah"]))
                            has_dimensions = (("dimensions" in text_lower or "abmessungen" in text_lower or 
                                             "weight" in text_lower or "gewicht" in text_lower) and 
                                            any(unit in text_lower for unit in ["mm", "kg", "inches", "lbs", "pounds", "g"]))
                            
                            is_technical_chunk = (has_performance or has_core_specs or has_display or has_battery or has_dimensions)
                            # Accept if product in text OR if it's a technical chunk (likely from the correct model/gen doc)
                            product_in_chunk = has_product_in_text or is_technical_chunk
                        else:
                            # Without generation, require explicit product mention
                            product_in_chunk = has_product_in_text
                        
                        # If product filtering is active but chunk doesn't match, skip it (unless similarity is very high)
                        # For queries with generation specified and PERFORMANCE chunks, be more lenient
                        threshold = 0.25 if (target_gen is not None and "performance" in text_lower) else 0.3
                        if not product_in_chunk and similarity < threshold:
                            logger.debug(f"Filtered out result {i} - product mismatch (target: {target_product}, similarity: {similarity})")
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
            if is_spec_query:
                retrieved_docs.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        else:
            logger.warning(f"No results returned from ChromaDB for user {user_id}")
        
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
        
        if is_spec_query:
            # For spec questions, get even more candidates to ensure we find technical chunks
            initial_k = max(target_k * 8, 40)  # Get 8x more for spec questions to ensure technical chunks are found
            logger.info(f"Specification query detected, retrieving {initial_k} candidates")
        else:
            initial_k = max(target_k * 3, 10)  # Get 3x more for reranking, minimum 10
        
        retrieved = self.retrieve(user_id, query, n_results=initial_k)
        
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


