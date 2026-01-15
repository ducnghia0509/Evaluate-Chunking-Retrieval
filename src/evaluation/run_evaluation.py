"""
Evaluation Dashboard with Gradio
Test different chunking strategies and retrieval methods
"""

import json
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
import sys
import plotly.graph_objects as go
import plotly.express as px

# Add retrieval path
sys.path.append(str(Path(__file__).parent.parent / "retrieval"))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch

# Import retrieval strategies
try:
    from parent_retrieval import ParentRetrieval
    PARENT_RETRIEVAL_AVAILABLE = True
except ImportError:
    PARENT_RETRIEVAL_AVAILABLE = False
    print("⚠️ ParentRetrieval not available")

try:
    from dense_retrieval import DenseRetrieval
    DENSE_RETRIEVAL_AVAILABLE = True
except ImportError:
    DENSE_RETRIEVAL_AVAILABLE = False
    print("⚠️ DenseRetrieval not available")

try:
    from sparse_retrieval import SparseRetrieval
    SPARSE_RETRIEVAL_AVAILABLE = True
except ImportError:
    SPARSE_RETRIEVAL_AVAILABLE = False
    print("⚠️ SparseRetrieval not available")

try:
    from hybrid_retrieval import HybridRetrieval
    HYBRID_RETRIEVAL_AVAILABLE = True
except ImportError:
    HYBRID_RETRIEVAL_AVAILABLE = False
    print("⚠️ HybridRetrieval not available")

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "evaluate"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
# EVALUATION_FILE = "../../evaluation.json"
EVALUATION_FOLDER = "../../splitted_by_category"
RESULTS_DIR = "../../Evaluate/results"


class EvaluationSystem:
    """Evaluation system for RAG with different chunking and retrieval strategies"""
    
    def __init__(self, evaluation_file: str = None):
        # Initialize Qdrant client
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Load embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
        
        # Store evaluation file path
        self.evaluation_file = evaluation_file
        
        # Load evaluation queries
        self.load_evaluation_queries()
        
        # Store comparison results
        self.comparison_results = {}
        
        # Initialize parent retrieval if available
        if PARENT_RETRIEVAL_AVAILABLE:
            try:
                self.parent_retrieval = ParentRetrieval()
                print("✓ ParentRetrieval initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize ParentRetrieval: {e}")
                self.parent_retrieval = None
        else:
            self.parent_retrieval = None
        
        # Initialize dense retrieval if available
        if DENSE_RETRIEVAL_AVAILABLE:
            try:
                self.dense_retrieval = DenseRetrieval()
                print("✓ DenseRetrieval initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize DenseRetrieval: {e}")
                self.dense_retrieval = None
        else:
            self.dense_retrieval = None
        
        # Initialize sparse retrieval if available
        if SPARSE_RETRIEVAL_AVAILABLE:
            try:
                self.sparse_retrieval = SparseRetrieval()
                print("✓ SparseRetrieval initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize SparseRetrieval: {e}")
                self.sparse_retrieval = None
        else:
            self.sparse_retrieval = None
        
        # Initialize hybrid retrieval if available
        if HYBRID_RETRIEVAL_AVAILABLE:
            try:
                self.hybrid_retrieval = HybridRetrieval()
                print("✓ HybridRetrieval initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize HybridRetrieval: {e}")
                self.hybrid_retrieval = None
        else:
            self.hybrid_retrieval = None
    
    def load_evaluation_queries(self):
        """Load evaluation queries from JSON file"""
        if self.evaluation_file:
            eval_path = Path(self.evaluation_file)
            if eval_path.exists():
                with open(eval_path, 'r', encoding='utf-8') as f:
                    self.queries = json.load(f)
                print(f"✓ Loaded {len(self.queries)} queries from {eval_path.name}")
            else:
                print(f"⚠️ Evaluation file not found: {eval_path}")
                self.queries = []
        else:
            print("⚠️ No evaluation file selected")
            self.queries = []
    
    def enrich_parent_child_context(self, chunks: List[Dict]) -> List[Dict]:
        """
        Enrich parent-child chunks with full context
        - For child chunks: add parent context
        - For parent chunks: add child contexts
        
        Args:
            chunks: Retrieved chunks
        
        Returns:
            Enriched chunks with full context
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        enriched_chunks = []
        
        for chunk in chunks:
            # Check both chunk_type (direct field) and metadata.role (nested field)
            chunk_type = chunk.get('chunk_type') or chunk.get('metadata', {}).get('role')
            
            if chunk_type == 'child':
                # Get parent context
                parent_id = chunk.get('metadata', {}).get('parent_id')
                if parent_id:
                    try:
                        # Search for parent by chunk_id
                        parent_filter = Filter(
                            must=[
                                FieldCondition(
                                    key="chunk_id",
                                    match=MatchValue(value=parent_id)
                                )
                            ]
                        )
                        
                        parent_results = self.client.scroll(
                            collection_name=COLLECTION_NAME,
                            scroll_filter=parent_filter,
                            limit=1,
                            with_payload=True
                        )[0]
                        
                        if parent_results:
                            parent_text = parent_results[0].payload.get('text', '')
                            parent_tokens = parent_results[0].payload.get('metadata', {}).get('token_count', 0)
                            
                            # Create enriched chunk
                            enriched_chunk = chunk.copy()
                            enriched_chunk['enriched_text'] = f"[PARENT CONTEXT]\n{parent_text}\n\n[CHILD CHUNK]\n{chunk['text']}"
                            enriched_chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0) + parent_tokens
                            enriched_chunk['context_type'] = 'child_with_parent'
                            enriched_chunks.append(enriched_chunk)
                        else:
                            # No parent found, use original
                            chunk['enriched_text'] = chunk['text']
                            chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0)
                            chunk['context_type'] = 'child_only'
                            enriched_chunks.append(chunk)
                    except Exception as e:
                        print(f"Warning: Could not fetch parent for {parent_id}: {e}")
                        chunk['enriched_text'] = chunk['text']
                        chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0)
                        chunk['context_type'] = 'child_only'
                        enriched_chunks.append(chunk)
                else:
                    chunk['enriched_text'] = chunk['text']
                    chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0)
                    chunk['context_type'] = 'child_only'
                    enriched_chunks.append(chunk)
                    
            elif chunk_type == 'parent':
                # Get all child contexts
                chunk_id = chunk.get('chunk_id')
                try:
                    # Search for children by parent_id
                    children_filter = Filter(
                        must=[
                            FieldCondition(
                                key="metadata.parent_id",
                                match=MatchValue(value=chunk_id)
                            )
                        ]
                    )
                    
                    children_results = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=children_filter,
                        limit=100,
                        with_payload=True
                    )[0]
                    
                    if children_results:
                        children_texts = [c.payload.get('text', '') for c in children_results]
                        children_tokens = sum(c.payload.get('metadata', {}).get('token_count', 0) for c in children_results)
                        
                        # Create enriched chunk
                        enriched_chunk = chunk.copy()
                        enriched_chunk['enriched_text'] = f"[PARENT CHUNK]\n{chunk['text']}\n\n[CHILD CHUNKS]\n" + "\n\n".join(children_texts)
                        enriched_chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0) + children_tokens
                        enriched_chunk['context_type'] = 'parent_with_children'
                        enriched_chunk['num_children'] = len(children_results)
                        enriched_chunks.append(enriched_chunk)
                    else:
                        chunk['enriched_text'] = chunk['text']
                        chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0)
                        chunk['context_type'] = 'parent_only'
                        enriched_chunks.append(chunk)
                except Exception as e:
                    print(f"Warning: Could not fetch children for {chunk_id}: {e}")
                    chunk['enriched_text'] = chunk['text']
                    chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0)
                    chunk['context_type'] = 'parent_only'
                    enriched_chunks.append(chunk)
            else:
                # Standard chunk, no enrichment
                chunk['enriched_text'] = chunk['text']
                chunk['enriched_tokens'] = chunk.get('metadata', {}).get('token_count', 0)
                chunk['context_type'] = 'standard'
                enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    def search_qdrant(self, query: str, chunking_strategy: str, top_k: int = 10, 
                      min_total_tokens: int = None, max_total_tokens: int = None, avg_chunk_tokens: int = 250) -> List[Dict]:
        """
        Search in Qdrant with chunking strategy filter and ensure minimum/maximum total tokens
        
        Args:
            query: Search query
            chunking_strategy: Chunking strategy to filter
            top_k: Number of results (initial)
            min_total_tokens: Minimum total tokens to retrieve (if None, calculated as k * avg_chunk_tokens)
            max_total_tokens: Maximum total tokens to retrieve (if set, will stop when reached)
            avg_chunk_tokens: Average tokens per chunk for calculating min_total_tokens
        
        Returns:
            List of retrieved chunks
        """
        # Calculate min_total_tokens if not provided
        if min_total_tokens is None:
            min_total_tokens = top_k * avg_chunk_tokens
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        # Search with filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="chunking_strategy",
                    match=MatchValue(value=chunking_strategy)
                )
            ]
        )
        
        # For parent_child strategy, only retrieve child chunks (smaller, better for retrieval)
        # Then enrich with parent context later
        if chunking_strategy == "parent_child":
            # Try metadata.role first, fallback to chunk_role if needed
            try:
                # First attempt with metadata.role
                search_filter.must.append(
                    FieldCondition(
                        key="metadata.role",
                        match=MatchValue(value="child")
                    )
                )
            except:
                # Fallback: try chunk_role
                search_filter.must.append(
                    FieldCondition(
                        key="chunk_role",
                        match=MatchValue(value="child")
                    )
                )
        
        # Start with initial top_k
        current_limit = top_k
        chunks = []
        total_tokens = 0
        previous_count = 0
        
        try:
            while True:
                results = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_embedding,
                    limit=current_limit,
                    query_filter=search_filter,
                    with_payload=True
                ).points
                
                # Convert to dict
                chunks = []
                total_tokens = 0
                
                for hit in results:
                    chunk_data = {
                        'chunk_id': hit.payload['chunk_id'],
                        'text': hit.payload['text'],
                        'score': float(hit.score),
                        'source_file': hit.payload.get('source_file', ''),
                        'category': hit.payload.get('category', ''),
                        'metadata': hit.payload.get('metadata', {})
                    }
                    chunks.append(chunk_data)
                    
                    # Count tokens
                    token_count = hit.payload.get('metadata', {}).get('token_count', 0)
                    if token_count > 0:
                        total_tokens += token_count
                    else:
                        # Estimate if not available (rough estimate: 1 token ≈ 4 chars)
                        total_tokens += len(hit.payload['text']) // 4
                
                # Check if we have enough tokens
                if total_tokens >= min_total_tokens:
                    # We have enough tokens
                    break
                
                # Check if we exceeded max tokens (if set)
                if max_total_tokens and total_tokens >= max_total_tokens:
                    # Trim chunks to fit max_total_tokens
                    trimmed_chunks = []
                    cumulative_tokens = 0
                    for chunk in chunks:
                        chunk_tokens = chunk.get('metadata', {}).get('token_count', 0) or len(chunk['text']) // 4
                        if cumulative_tokens + chunk_tokens <= max_total_tokens:
                            trimmed_chunks.append(chunk)
                            cumulative_tokens += chunk_tokens
                        else:
                            break
                    chunks = trimmed_chunks
                    total_tokens = cumulative_tokens
                    print(f"✂️ Trimmed to {len(chunks)} chunks ({total_tokens} tokens) to fit max_total_tokens={max_total_tokens}")
                    break
                
                # Check if we got more results than previous iteration
                if len(results) == previous_count:
                    # No more results available, even with increased limit
                    print(f"⚠️ Could only retrieve {len(results)} chunks ({total_tokens} tokens) - "
                          f"target was {min_total_tokens} tokens. May indicate limited data in collection.")
                    break
                
                # Check if we reached the result count (might have more with higher limit)
                if len(results) < current_limit:
                    # We got fewer results than requested - this is max available
                    print(f"⚠️ Retrieved all available {len(results)} chunks ({total_tokens} tokens) - "
                          f"target was {min_total_tokens} tokens.")
                    break
                
                # Need more chunks - increase limit
                previous_count = len(results)
                current_limit = int(current_limit * 1.5)
                
                # Safety limit to prevent infinite loop
                if current_limit > 100:
                    print(f"⚠️ Reached safety limit (100 chunks), stopping at {len(results)} chunks ({total_tokens} tokens)")
                    break
            
            if len(chunks) > 0:
                # Add summary to first chunk for logging
                chunks[0]['_retrieval_summary'] = {
                    'total_chunks': len(chunks),
                    'total_tokens': total_tokens,
                    'min_total_tokens_target': min_total_tokens,
                    'avg_tokens_per_chunk': total_tokens / len(chunks) if chunks else 0
                }
        
        except Exception as e:
            print(f"⚠️ Error querying Qdrant for strategy '{chunking_strategy}': {e}")
            import traceback
            traceback.print_exc()
            return []
        
        if len(chunks) == 0:
            print(f"⚠️ No chunks found for strategy: {chunking_strategy}")
            print(f"   Filter used: chunking_strategy={chunking_strategy}")
            if chunking_strategy == "parent_child":
                print(f"   Also filtered by: metadata.role=child")
            return []
        
        # Enrich parent-child chunks with full context
        if chunking_strategy == "parent_child":
            print(f"✓ Retrieved {len(chunks)} chunks ({total_tokens} tokens)")
            # Debug: Check what we actually retrieved
            if chunks:
                sample_ids = [c['chunk_id'] for c in chunks[:3]]
                print(f"  Sample chunk IDs BEFORE enrichment: {sample_ids}")
            
            print(f"  Enriching with parent/child context...")
            chunks = self.enrich_parent_child_context(chunks)
            
            # Debug: Check after enrichment
            if chunks:
                sample_ids_after = [c['chunk_id'] for c in chunks[:3]]
                print(f"  Sample chunk IDs AFTER enrichment: {sample_ids_after}")
            
            # Recalculate total tokens after enrichment
            enriched_total_tokens = sum(c.get('enriched_tokens', 0) for c in chunks)
            print(f"✓ Enriched to {enriched_total_tokens} tokens total (avg {enriched_total_tokens/len(chunks):.0f} per chunk)")
            
            # Update summary with enriched tokens
            if chunks and '_retrieval_summary' in chunks[0]:
                chunks[0]['_retrieval_summary']['total_tokens_enriched'] = enriched_total_tokens
                chunks[0]['_retrieval_summary']['avg_tokens_per_chunk_enriched'] = enriched_total_tokens / len(chunks)
        else:
            print(f"✓ Retrieved {len(chunks)} chunks ({total_tokens} tokens) for query with strategy '{chunking_strategy}'")
        
        return chunks
    
    def get_ground_truth_chunks_text(self, relevant_chunks: List[Dict], chunking_strategy: str) -> Dict[str, Dict]:
        """
        Get ground truth chunks with their text from Qdrant
        Looks for chunks in ANY strategy (ground truth may be from different strategy)
        
        Args:
            relevant_chunks: List of ground truth chunks with chunk_id
            chunking_strategy: Current chunking strategy being evaluated (for info only)
        
        Returns:
            Dict mapping chunk_id to chunk data with text
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        ground_truth_map = {}
        
        for rel_chunk in relevant_chunks:
            chunk_id = rel_chunk['chunk_id']
            
            # Try to find this chunk in Qdrant (any strategy)
            try:
                # Search without strategy filter to find chunk in any strategy
                results = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="chunk_id",
                                match=MatchValue(value=chunk_id)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True
                )[0]
                
                if results:
                    chunk_data = results[0].payload
                    ground_truth_map[chunk_id] = {
                        'text': chunk_data.get('text', ''),
                        'source_file': chunk_data.get('source_file', ''),
                        'relevance': rel_chunk.get('relevance', 'high'),
                        'strategy': chunk_data.get('chunking_strategy', 'unknown')
                    }
                else:
                    # Chunk not found
                    ground_truth_map[chunk_id] = {
                        'text': '',
                        'source_file': '',
                        'relevance': rel_chunk.get('relevance', 'high'),
                        'not_found': True
                    }
            except Exception as e:
                print(f"Warning: Could not fetch ground truth chunk {chunk_id}: {e}")
                ground_truth_map[chunk_id] = {
                    'text': '',
                    'source_file': '',
                    'relevance': rel_chunk.get('relevance', 'high'),
                    'error': str(e)
                }
        
        return ground_truth_map
    
    def calculate_metrics(self, retrieved_chunks: List[Dict], relevant_chunks: List[Dict], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Calculate evaluation metrics with flexible matching
        
        Args:
            retrieved_chunks: List of retrieved chunks (ordered by score)
            relevant_chunks: List of ground truth relevant chunks
            k_values: k values for Recall@k
        
        Returns:
            Dictionary of metrics
        """
        if not retrieved_chunks or not relevant_chunks:
            # Return zero metrics
            metrics = {}
            for k in k_values:
                metrics[f'recall@{k}'] = 0.0
                metrics[f'precision@{k}'] = 0.0
                metrics[f'ndcg@{k}'] = 0.0
            metrics['mrr'] = 0.0
            return metrics
        
        # Get ground truth chunk texts
        chunking_strategy = retrieved_chunks[0].get('metadata', {}).get('strategy', 'unknown')
        if not chunking_strategy or chunking_strategy == 'unknown':
            # Try to infer from chunk_id pattern
            chunk_id = retrieved_chunks[0].get('chunk_id', '')
            if '_parent_' in chunk_id or '_child_' in chunk_id:
                chunking_strategy = 'parent_child'
            elif '_chunk_' in chunk_id:
                chunking_strategy = 'fixed'  # or other
        
        ground_truth_map = self.get_ground_truth_chunks_text(relevant_chunks, chunking_strategy)
        
        # Build flexible matching
        # For each retrieved chunk, check if it matches any ground truth
        retrieved_matches = []  # List of (retrieved_chunk, matched_gt_id, relevance)
        
        # Debug info
        match_debug = []
        
        # Determine if we're matching parent-child strategy
        is_parent_child = False
        if retrieved_chunks:
            first_chunk_id = retrieved_chunks[0].get('chunk_id', '')
            is_parent_child = '_parent_' in first_chunk_id or '_child_' in first_chunk_id
        
        for ret_chunk in retrieved_chunks:
            ret_id = ret_chunk['chunk_id']
            
            # Get text for matching - handle different chunk types differently
            is_parent_chunk = '_parent_' in ret_id
            is_child_chunk = '_child_' in ret_id
            
            if is_parent_chunk:
                # For parent chunks (from parent retrieval), use full text
                # Parent text should contain the ground truth chunks
                ret_text = ret_chunk.get('enriched_text', '') or ret_chunk.get('text', '')
            elif is_child_chunk:
                # For child chunks, use original child text (not enriched)
                # Because enriched includes parent which makes it too large
                ret_text = ret_chunk.get('text', '')
            else:
                # For other strategies
                ret_text = ret_chunk.get('text', '')
            
            ret_source = ret_chunk.get('source_file', '')
            
            matched = False
            best_match = None
            best_overlap = 0.0
            
            # Strategy 1: Exact chunk_id match (same strategy)
            if ret_id in ground_truth_map:
                relevance = ground_truth_map[ret_id]['relevance']
                retrieved_matches.append((ret_chunk, ret_id, relevance))
                matched = True
                match_debug.append(f"Exact match: {ret_id}")
            else:
                # Strategy 2: Text overlap for cross-strategy matching
                # Check if retrieved chunk overlaps with any ground truth chunk
                for gt_id, gt_data in ground_truth_map.items():
                    if 'not_found' in gt_data or 'error' in gt_data:
                        continue
                    
                    gt_text = gt_data.get('text', '')
                    gt_source = gt_data.get('source_file', '')
                    
                    # Must be from same source file
                    if not gt_source or not ret_source:
                        continue
                    if gt_source != ret_source:
                        continue
                    
                    if not gt_text or not ret_text:
                        continue
                    
                    # Calculate text overlap
                    ret_text_clean = ret_text.lower().strip()
                    gt_text_clean = gt_text.lower().strip()
                    
                    # Calculate word overlap
                    ret_words = set(ret_text_clean.split())
                    gt_words = set(gt_text_clean.split())
                    
                    if ret_words and gt_words:
                        intersection = len(ret_words & gt_words)
                        union = len(ret_words | gt_words)
                        jaccard = intersection / union if union > 0 else 0.0
                        
                        # Containment: what % of GT words are in retrieved chunk
                        # This is key for parent chunks (large) containing ground truth (small)
                        containment = intersection / len(gt_words) if len(gt_words) > 0 else 0.0
                        
                        # For parent chunks, prioritize containment over jaccard
                        # Because parent is much larger, jaccard will be low even if GT is fully contained
                        if is_parent_chunk:
                            overlap_ratio = containment  # Use containment primarily
                            # Boost if GT text is substring of parent
                            if gt_text_clean in ret_text_clean:
                                overlap_ratio = max(overlap_ratio, 0.95)
                        else:
                            # For other chunks, use balanced approach
                            overlap_ratio = max(jaccard, containment)
                            
                            # Check substring match
                            if gt_text_clean in ret_text_clean or ret_text_clean in gt_text_clean:
                                overlap_ratio = max(overlap_ratio, 0.90)
                        
                        # Track best match
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_match = (gt_id, gt_data['relevance'])
                
                # Adaptive threshold based on chunk type
                if is_parent_chunk:
                    # Lower threshold for parent chunks since they're much larger
                    # Even 20% containment means significant overlap
                    match_threshold = 0.2  # Lowered from 0.30 to 0.20
                elif is_child_chunk:
                    # Normal threshold for child chunks
                    match_threshold = 0.1
                else:
                    # Normal threshold for other strategies
                    match_threshold = 0.1
                
                if best_match and best_overlap > match_threshold:
                    relevance = best_match[1]
                    retrieved_matches.append((ret_chunk, best_match[0], relevance))
                    matched = True
                    match_type = "parent containment" if is_parent_chunk else "text overlap"
                    match_debug.append(f"{match_type} match: {ret_id} -> {best_match[0]} (overlap: {best_overlap:.2%})")
                else:
                    # No match found
                    retrieved_matches.append((ret_chunk, None, None))
                    if best_overlap > 0:
                        match_debug.append(f"No match: {ret_id} (best overlap: {best_overlap:.2%}, threshold: {match_threshold:.0%})")
                    else:
                        match_debug.append(f"No match: {ret_id} (no overlap)")
                for gt_id, gt_data in ground_truth_map.items():
                    if 'not_found' in gt_data or 'error' in gt_data:
                        continue
                    
                    gt_text = gt_data.get('text', '')
                    gt_source = gt_data.get('source_file', '')
                    
                    # Must be from same source file
                    if not gt_source or not ret_source:
                        continue
                    if gt_source != ret_source:
                        continue
                    
                    if not gt_text or not ret_text:
                        continue
                    
                    # Calculate text overlap
                    ret_text_clean = ret_text.lower().strip()
                    gt_text_clean = gt_text.lower().strip()
                    
                    # Calculate word overlap
                    ret_words = set(ret_text_clean.split())
                    gt_words = set(gt_text_clean.split())
                    
                    if ret_words and gt_words:
                        intersection = len(ret_words & gt_words)
                        union = len(ret_words | gt_words)
                        jaccard = intersection / union if union > 0 else 0.0
                        
                        # Containment: what % of GT words are in retrieved chunk
                        # This is key for parent chunks (large) containing ground truth (small)
                        containment = intersection / len(gt_words) if len(gt_words) > 0 else 0.0
                        
                        # For parent chunks, prioritize containment over jaccard
                        # Because parent is much larger, jaccard will be low even if GT is fully contained
                        if is_parent_chunk:  # Use is_parent_chunk from outer scope
                            overlap_ratio = containment  # Use containment primarily
                            # Boost if GT text is substring of parent
                            if gt_text_clean in ret_text_clean:
                                overlap_ratio = max(overlap_ratio, 0.95)
                        else:
                            # For other chunks, use balanced approach
                            overlap_ratio = max(jaccard, containment)
                            
                            # Check substring match
                            if gt_text_clean in ret_text_clean or ret_text_clean in gt_text_clean:
                                overlap_ratio = max(overlap_ratio, 0.90)
                        
                        # Track best match
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_match = (gt_id, gt_data['relevance'])
                
                # Adaptive threshold based on chunk type (use is_parent_chunk from outer scope)
                if is_parent_chunk:
                    # Lower threshold for parent chunks since they're much larger
                    # Even 20% containment means significant overlap
                    match_threshold = 0.2
                elif is_child_chunk:
                    # Normal threshold for child chunks
                    match_threshold = 0.1
                else:
                    # Normal threshold for other strategies
                    match_threshold = 0.1
                
                if best_match and best_overlap > match_threshold:
                    relevance = best_match[1]
                    retrieved_matches.append((ret_chunk, best_match[0], relevance))
                    matched = True
                    match_type = "parent containment" if is_parent_chunk else "text overlap"
                    match_debug.append(f"{match_type} match: {ret_id} -> {best_match[0]} (overlap: {best_overlap:.2%})")
                else:
                    # No match found
                    retrieved_matches.append((ret_chunk, None, None))
                    if best_overlap > 0:
                        match_debug.append(f"No match: {ret_id} (best overlap: {best_overlap:.2%}, threshold: {match_threshold:.0%})")
                    else:
                        match_debug.append(f"No match: {ret_id} (no overlap)")
        
        # Extract IDs for metrics calculation
        retrieved_ids = [match[1] if match[1] else f"_no_match_{i}" for i, match in enumerate(retrieved_matches)]
        relevant_ids = {gt_id: gt_data['relevance'] for gt_id, gt_data in ground_truth_map.items() 
                       if 'not_found' not in gt_data and 'error' not in gt_data}
        
        # Debug: Show matching statistics
        num_matches = sum(1 for match in retrieved_matches if match[1] is not None)
        if num_matches == 0 and len(retrieved_chunks) > 0 and len(relevant_chunks) > 0:
            print(f"\n  [DEBUG] Matching failed:")
            print(f"    Total matches: {num_matches}/{len(retrieved_matches)}")
            print(f"    First 3 match attempts:")
            for i, debug_msg in enumerate(match_debug[:3]):
                print(f"      {i+1}. {debug_msg}")
        elif num_matches > 0:
            print(f"  ✓ Found {num_matches} matches out of {len(retrieved_matches)} retrieved chunks")
        
        metrics = {}
        
        # Recall@k
        for k in k_values:
            retrieved_k = set(retrieved_ids[:k])
            relevant_set = set(relevant_ids.keys())
            
            if len(relevant_set) > 0:
                recall = len(retrieved_k & relevant_set) / len(relevant_set)
            else:
                recall = 0.0
            
            metrics[f'recall@{k}'] = recall
        
        # Precision@k
        for k in k_values:
            retrieved_k = set(retrieved_ids[:k])
            relevant_set = set(relevant_ids.keys())
            
            if k > 0:
                precision = len(retrieved_k & relevant_set) / k
            else:
                precision = 0.0
            
            metrics[f'precision@{k}'] = precision
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr
        
        # NDCG@k (simplified version)
        # Relevance score: high=3, medium=2, low=1
        relevance_map = {'high': 3, 'medium': 2, 'low': 1}
        
        for k in k_values:
            dcg = 0.0
            for i in range(min(k, len(retrieved_ids))):
                chunk_id = retrieved_ids[i]
                if chunk_id in relevant_ids:
                    rel = relevance_map.get(relevant_ids[chunk_id], 1)
                    dcg += rel / (i + 1)
            
            # Ideal DCG
            ideal_rels = sorted([relevance_map.get(v, 1) for v in relevant_ids.values()], reverse=True)
            idcg = sum([rel / (i + 1) for i, rel in enumerate(ideal_rels[:k])])
            
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            metrics[f'ndcg@{k}'] = ndcg
        
        return metrics
    
    def run_evaluation(self, chunking_strategy: str, retrieval_method: str, 
                      top_k: int = 10, min_total_tokens: int = None, max_total_tokens: int = None,
                      progress=gr.Progress()) -> Tuple[str, pd.DataFrame, str]:
        if not self.queries:
            return "No evaluation queries loaded!", pd.DataFrame(), "{}", self.generate_comparison_heatmap()
        
        results = []
        all_metrics = []
        queries_with_no_results = 0
        total_chunks_retrieved = 0
        total_tokens_retrieved = 0
        
        for i, query_item in enumerate(progress.tqdm(self.queries, desc="Evaluating")):
            query_id = query_item['id']
            query_text = query_item['question']
            relevant_chunks = query_item.get('relevant_chunks', [])
            
            # Choose retrieval method
            if retrieval_method == "parent" and chunking_strategy == "parent_child":
                # Use ParentRetrieval for parent-child strategy
                if self.parent_retrieval:
                    try:
                        parent_results = self.parent_retrieval.search(query_text, top_k=top_k)
                        # Convert to standard format
                        retrieved_chunks = []
                        for pr in parent_results:
                            # Need to get source_file from the parent chunk
                            parent_id = pr['parent_id']
                            
                            # Fetch parent chunk details
                            try:
                                from qdrant_client.models import Filter, FieldCondition, MatchValue
                                parent_filter = Filter(
                                    must=[
                                        FieldCondition(
                                            key="chunk_id",
                                            match=MatchValue(value=parent_id)
                                        )
                                    ]
                                )
                                
                                parent_details = self.client.scroll(
                                    collection_name=COLLECTION_NAME,
                                    scroll_filter=parent_filter,
                                    limit=1,
                                    with_payload=True
                                )[0]
                                
                                if parent_details:
                                    parent_payload = parent_details[0].payload
                                    source_file = parent_payload.get('source_file', '')
                                    category = parent_payload.get('category', '')
                                else:
                                    source_file = ''
                                    category = ''
                            except:
                                source_file = ''
                                category = ''
                            
                            retrieved_chunks.append({
                                'chunk_id': parent_id,
                                'text': pr['context'],
                                'score': pr['retrieval_score'],
                                'source_file': source_file,
                                'category': category,
                                'metadata': {
                                    'num_child_hits': pr['num_child_hits'],
                                    'retrieval_method': 'parent',
                                    'token_count': int(len(pr['context'].split()) * 1.3)  # Rough estimate
                                },
                                'enriched_text': pr['context'],
                                'enriched_tokens': int(len(pr['context'].split()) * 1.3)
                            })
                    except Exception as e:
                        print(f"⚠️ Error using ParentRetrieval: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to dense search
                        retrieved_chunks = self.search_qdrant(
                            query_text, 
                            chunking_strategy, 
                            top_k=top_k,
                            min_total_tokens=min_total_tokens,
                            max_total_tokens=max_total_tokens
                        )
                else:
                    print("⚠️ ParentRetrieval not available, using dense search")
                    retrieved_chunks = self.search_qdrant(
                        query_text, 
                        chunking_strategy, 
                        top_k=top_k,
                        min_total_tokens=min_total_tokens,
                        max_total_tokens=max_total_tokens
                    )
            elif retrieval_method == "dense":
                # Use DenseRetrieval
                if self.dense_retrieval:
                    try:
                        dense_results = self.dense_retrieval.search(
                            query_text, 
                            chunking_strategy=chunking_strategy,
                            top_k=top_k
                        )
                        retrieved_chunks = dense_results
                    except Exception as e:
                        print(f"⚠️ Error using DenseRetrieval: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to search_qdrant
                        retrieved_chunks = self.search_qdrant(
                            query_text, 
                            chunking_strategy, 
                            top_k=top_k,
                            min_total_tokens=min_total_tokens,
                            max_total_tokens=max_total_tokens
                        )
                else:
                    print("⚠️ DenseRetrieval not available, using search_qdrant")
                    retrieved_chunks = self.search_qdrant(
                        query_text, 
                        chunking_strategy, 
                        top_k=top_k,
                        min_total_tokens=min_total_tokens,
                        max_total_tokens=max_total_tokens
                    )
            elif retrieval_method == "sparse":
                # Use SparseRetrieval
                if self.sparse_retrieval:
                    try:
                        sparse_results = self.sparse_retrieval.search(
                            query_text, 
                            chunking_strategy=chunking_strategy,
                            top_k=top_k
                        )
                        retrieved_chunks = sparse_results
                    except Exception as e:
                        print(f"⚠️ Error using SparseRetrieval: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to search_qdrant
                        retrieved_chunks = self.search_qdrant(
                            query_text, 
                            chunking_strategy, 
                            top_k=top_k,
                            min_total_tokens=min_total_tokens,
                            max_total_tokens=max_total_tokens
                        )
                else:
                    print("⚠️ SparseRetrieval not available, using search_qdrant")
                    retrieved_chunks = self.search_qdrant(
                        query_text, 
                        chunking_strategy, 
                        top_k=top_k,
                        min_total_tokens=min_total_tokens,
                        max_total_tokens=max_total_tokens
                    )
            elif retrieval_method == "hybrid":
                # Use HybridRetrieval
                if self.hybrid_retrieval:
                    try:
                        hybrid_results = self.hybrid_retrieval.search(
                            query_text, 
                            chunking_strategy=chunking_strategy,
                            top_k=top_k
                        )
                        retrieved_chunks = hybrid_results
                    except Exception as e:
                        print(f"⚠️ Error using HybridRetrieval: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to search_qdrant
                        retrieved_chunks = self.search_qdrant(
                            query_text, 
                            chunking_strategy, 
                            top_k=top_k,
                            min_total_tokens=min_total_tokens,
                            max_total_tokens=max_total_tokens
                        )
                else:
                    print("⚠️ HybridRetrieval not available, using search_qdrant")
                    retrieved_chunks = self.search_qdrant(
                        query_text, 
                        chunking_strategy, 
                        top_k=top_k,
                        min_total_tokens=min_total_tokens,
                        max_total_tokens=max_total_tokens
                    )
            else:
                # Use standard dense search (fallback)
                retrieved_chunks = self.search_qdrant(
                    query_text, 
                    chunking_strategy, 
                    top_k=top_k,
                    min_total_tokens=min_total_tokens,
                    max_total_tokens=max_total_tokens
                )
            
            if len(retrieved_chunks) == 0:
                queries_with_no_results += 1
            else:
                total_chunks_retrieved += len(retrieved_chunks)
                
                # Extract retrieval summary if available
                if '_retrieval_summary' in retrieved_chunks[0]:
                    summary = retrieved_chunks[0]['_retrieval_summary']
                    total_tokens_retrieved += summary['total_tokens']
                else:
                    # Fallback: calculate from enriched_tokens or estimate
                    for chunk in retrieved_chunks:
                        # Try enriched_tokens first, then metadata token_count, then estimate
                        tokens = chunk.get('enriched_tokens') or \
                                chunk.get('metadata', {}).get('token_count') or \
                                len(chunk.get('text', '')) // 4
                        total_tokens_retrieved += tokens
            
            # Calculate metrics
            metrics = self.calculate_metrics(retrieved_chunks, relevant_chunks)
            
            # Debug: Check if any matches found
            if all(m == 0 for m in metrics.values()):
                print(f"⚠️ Query {query_id}: No matches found!")
                if retrieved_chunks:
                    print(f"   Retrieved chunk IDs (first 3): {[c['chunk_id'] for c in retrieved_chunks[:3]]}")
                if relevant_chunks:
                    print(f"   Expected chunk IDs: {[c['chunk_id'] for c in relevant_chunks]}")
            
            # Store result
            result = {
                'query_id': query_id,
                'question': query_item.get('question', ''),
                'query': query_text,
                'chunking_strategy': chunking_strategy,
                'retrieval_method': retrieval_method,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'retrieved_chunks': retrieved_chunks[:5],  # Top 5 for display
                'retrieval_stats': {
                    'total_chunks': len(retrieved_chunks),
                    'total_tokens': retrieved_chunks[0].get('_retrieval_summary', {}).get('total_tokens', 0) if retrieved_chunks else 0,
                    'avg_tokens_per_chunk': retrieved_chunks[0].get('_retrieval_summary', {}).get('avg_tokens_per_chunk', 0) if retrieved_chunks else 0
                },
                'relevant_chunks': relevant_chunks
            }
            results.append(result)
            all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        if all_metrics:
            metric_keys = all_metrics[0].keys()
            for key in metric_keys:
                avg_metrics[key] = sum([m[key] for m in all_metrics]) / len(all_metrics)
        
        # Create summary
        summary = f"**Evaluation Summary**\n\n"
        summary += f"- Chunking Strategy: **{chunking_strategy}**\n"
        summary += f"- Retrieval Method: **{retrieval_method}**\n"
        summary += f"- Total Queries: **{len(self.queries)}**\n"
        summary += f"- Top-K: **{top_k}**\n"
        
        if min_total_tokens:
            summary += f"- Min Total Tokens: **{min_total_tokens}** (for fair comparison)\n"
        
        # Add retrieval statistics
        # avg_chunks_per_query = total_chunks_retrieved / len(self.queries) if self.queries else 0
        avg_tokens_per_query = total_tokens_retrieved / len(self.queries) if self.queries else 0
        
        summary += f"\n**Retrieval Statistics:**\n"
        # summary += f"- Avg chunks per query: **{avg_chunks_per_query:.2f}**\n"
        summary += f"- Avg tokens per query: **{avg_tokens_per_query:.2f}**\n"
        
        if queries_with_no_results > 0:
            summary += f"\n⚠️ **Queries with no results: {queries_with_no_results}/{len(self.queries)}**\n"
            summary += f"  - *This may indicate missing data in Qdrant for this chunking strategy*\n"
        
        summary += "\n**Average Metrics:**\n"
        for metric, value in avg_metrics.items():
            summary += f"- {metric}: **{value:.4f}**\n"
        
        # Create DataFrame for display
        df_data = []
        for r in results:
            row = {
                'Query ID': r['query_id'],
                'Question': r['question'][:50] + '...' if len(r['question']) > 50 else r['question'],
            }
            row.update({k: f"{v:.4f}" for k, v in r['metrics'].items()})
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save results
        output_dir = Path(RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"eval_{chunking_strategy}_{retrieval_method}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': avg_metrics,
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        detailed_results = json.dumps({
            'summary': avg_metrics,
            'results': results
        }, ensure_ascii=False, indent=2)
        
        # Store in comparison results (including sum tokens and chunks)
        strategy_key = f"{chunking_strategy}+{retrieval_method}"
        self.comparison_results[strategy_key] = {
            'metrics': avg_metrics,
            # 'sum_tokens': total_tokens_retrieved,
            # 'sum_chunks': total_chunks_retrieved,
            # 'avg_chunks_per_query': avg_chunks_per_query,
            'avg_tokens_per_query': avg_tokens_per_query,
            'timestamp': datetime.now().isoformat(),
            'top_k': top_k,
            'min_total_tokens': min_total_tokens,
            'max_total_tokens': max_total_tokens
        }
        
        # Generate comparison heatmap and strategy table
        comparison_fig = self.generate_comparison_heatmap()
        strategy_table = self.generate_strategy_summary_table()
        
        return summary, df, detailed_results, comparison_fig, strategy_table
    
    def run_all_evaluations(self, top_k: int = 10, min_total_tokens: int = None, 
                           max_total_tokens: int = None, progress=gr.Progress()) -> Tuple[str, pd.DataFrame, str]:
        """
        Run all combinations of chunking strategies and retrieval methods
        Parent retrieval only runs with parent_child chunking
        
        Args:
            top_k: Number of results to retrieve
            min_total_tokens: Minimum total tokens
            max_total_tokens: Maximum total tokens
            progress: Gradio progress tracker
        
        Returns:
            Summary text, comparison table, and detailed results
        """
        chunking_strategies = ["fixed", "structure_paragraph", "hierarchical", "parent_child"]
        all_retrieval_methods = ["dense", "hybrid"]
        
        # Calculate total combinations
        total_combinations = len(chunking_strategies) * len(all_retrieval_methods) + 1  # +1 for parent_child + parent
        
        results_summary = f"## 🚀 Running All Evaluations\n\n"
        results_summary += f"Total combinations: **{total_combinations}**\n\n"
        
        completed = 0
        
        for chunking_strategy in chunking_strategies:
            # Determine retrieval methods for this chunking strategy
            if chunking_strategy == "parent_child":
                retrieval_methods = all_retrieval_methods + ["parent"]
            else:
                retrieval_methods = all_retrieval_methods
            
            for retrieval_method in retrieval_methods:
                completed += 1
                
                # Update progress
                progress_pct = completed / total_combinations
                progress(progress_pct, desc=f"Evaluating {chunking_strategy} + {retrieval_method} ({completed}/{total_combinations})")
                
                print(f"\n{'='*80}")
                print(f"Running: {chunking_strategy} + {retrieval_method} ({completed}/{total_combinations})")
                print(f"{'='*80}\n")
                
                try:
                    # Run evaluation
                    summary, df, detailed, comparison_fig, strategy_table = self.run_evaluation(
                        chunking_strategy=chunking_strategy,
                        retrieval_method=retrieval_method,
                        top_k=top_k,
                        min_total_tokens=min_total_tokens,
                        max_total_tokens=max_total_tokens,
                        progress=progress
                    )
                    
                    results_summary += f"✅ **{chunking_strategy} + {retrieval_method}**: Completed\n"
                    
                except Exception as e:
                    results_summary += f"❌ **{chunking_strategy} + {retrieval_method}**: Error - {str(e)}\n"
                    print(f"Error in {chunking_strategy} + {retrieval_method}: {e}")
                    # traceback.print_exc()
        
        # Generate final comparison heatmap and table
        final_comparison_fig = self.generate_comparison_heatmap()
        final_strategy_table = self.generate_strategy_summary_table()
        
        results_summary += f"\n\n## ✨ All Evaluations Completed!\n\n"
        results_summary += f"Check the **Comparison Heatmap** and **Strategy Comparison Table** tabs for results.\n"
        
        detailed_results = json.dumps({
            'all_strategies': list(self.comparison_results.keys()),
            'total_combinations': total_combinations,
            'completed': completed
        }, ensure_ascii=False, indent=2)
        
        return results_summary, final_strategy_table, detailed_results, final_comparison_fig, final_strategy_table
    
    def generate_strategy_summary_table(self):
        """Generate summary table of all evaluated strategies"""
        if not self.comparison_results:
            return pd.DataFrame()
        
        # Create summary data
        summary_data = []
        for strategy_name, strategy_data in self.comparison_results.items():
            metrics = strategy_data.get('metrics', {})
            row = {
                'Strategy': strategy_name,
                # 'Total Tokens': int(strategy_data.get('sum_tokens', 0)),
                # 'Total Chunks': int(strategy_data.get('sum_chunks', 0)),
                'Avg Tokens/Query': f"{strategy_data.get('avg_tokens_per_query', 0):.2f}",
                'Recall@10': f"{metrics.get('recall@10', 0):.4f}",
                'Precision@10': f"{metrics.get('precision@10', 0):.4f}",
                'MRR': f"{metrics.get('mrr', 0):.4f}",
                'NDCG@10': f"{metrics.get('ndcg@10', 0):.4f}"
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def generate_comparison_heatmap(self):
        """Generate heatmap comparing all evaluated strategies"""
        if not self.comparison_results:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No comparison data yet. Run evaluations to see comparison.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                height=400,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Prepare data for heatmap
        strategies = list(self.comparison_results.keys())
        
        # Get all metrics from first strategy
        first_strategy = strategies[0]
        all_metrics = list(self.comparison_results[first_strategy]['metrics'].keys())
        
        # Add sum_tokens and sum_chunks to metrics
        all_metrics_extended = all_metrics + ['avg_tokens_per_query']
        
        # Create matrix: rows = metrics, columns = strategies
        data_matrix = []
        for metric in all_metrics_extended:
            row = []
            for strategy in strategies:
                if metric in ['avg_tokens_per_query']:
                    val = self.comparison_results[strategy].get(metric, 0)
                else:
                    val = self.comparison_results[strategy]['metrics'].get(metric, 0)
                row.append(val)
            data_matrix.append(row)
        
        # Format text for display
        text_matrix = []
        for i, metric in enumerate(all_metrics_extended):
            row = []
            for val in data_matrix[i]:
                if metric in ['sum_tokens', 'sum_chunks']:
                    row.append(f"{int(val)}")
                else:
                    row.append(f"{val:.4f}")
            text_matrix.append(row)
        
        # Normalize by row (each metric independently) for better color comparison
        normalized_matrix = []
        for i, row in enumerate(data_matrix):
            metric = all_metrics_extended[i]
            
            # Only avg_tokens_per_query: lower is better (inverse normalization)
            # All other metrics: higher is better (normal normalization)
            if metric == 'avg_tokens_per_query':
                min_val = min(row)
                max_val = max(row)
                if max_val > min_val:
                    # Inverse: lower values get higher scores (greener)
                    normalized_row = [1 - ((val - min_val) / (max_val - min_val)) for val in row]
                else:
                    normalized_row = [0.5] * len(row)
            else:
                # For all other metrics, higher is better (normal normalization)
                min_val = min(row)
                max_val = max(row)
                if max_val > min_val:
                    # Normal: higher values get higher scores (greener)
                    normalized_row = [(val - min_val) / (max_val - min_val) for val in row]
                else:
                    normalized_row = [0.5] * len(row)
            
            normalized_matrix.append(normalized_row)
        
        # Create heatmap with row-normalized colors
        fig = go.Figure(data=go.Heatmap(
            z=normalized_matrix,
            x=strategies,
            y=all_metrics_extended,
            colorscale='RdYlGn',  # Red-Yellow-Green
            text=text_matrix,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Relative Score<br>(per metric)"),
            hovertemplate='Strategy: %{x}<br>Metric: %{y}<br>Value: %{text}<extra></extra>',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title="Comparison Heatmap: Chunking + Retrieval Strategies (Click on column to see details)",
            xaxis_title="Strategy (Chunking + Retrieval)",
            yaxis_title="Metrics",
            height=700,
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    
    def get_strategy_details(self, evt: gr.SelectData):
        """Get detailed metrics for selected strategy from table"""
        if not evt:
            return "Click on a row in the strategy table to see detailed metrics."
        
        # Get selected row index
        row_index = evt.index[0] if isinstance(evt.index, list) else evt.index
        
        # Get strategy name from comparison results by index
        strategy_names = list(self.comparison_results.keys())
        
        if row_index >= len(strategy_names):
            return "Invalid selection."
        
        strategy_name = strategy_names[row_index]
        
        if strategy_name not in self.comparison_results:
            return f"No data found for strategy: {strategy_name}"
        
        # Get strategy data
        strategy_data = self.comparison_results[strategy_name]
        
        # Format detailed output
        details = f"## 📊 Detailed Metrics: {strategy_name}\n\n"
        
        # Evaluation info
        details += f"**Evaluation Information:**\n"
        details += f"- Timestamp: {strategy_data.get('timestamp', 'N/A')}\n"
        details += f"- Top-K: {strategy_data.get('top_k', 'N/A')}\n"
        details += f"- Min Total Tokens: {strategy_data.get('min_total_tokens', 'N/A')}\n"
        details += f"- Max Total Tokens: {strategy_data.get('max_total_tokens', 'N/A')}\n\n"
        
        # Retrieval statistics
        details += f"**Retrieval Statistics:**\n"
        # details += f"- Total Tokens: **{int(strategy_data.get('sum_tokens', 0))}**\n"
        # details += f"- Total Chunks: **{int(strategy_data.get('sum_chunks', 0))}**\n"
        details += f"- Avg Tokens/Query: **{strategy_data.get('avg_tokens_per_query', 0):.2f}**\n"
        # details += f"- Avg Chunks/Query: **{strategy_data.get('avg_chunks_per_query', 0):.2f}**\n\n"
        
        # Metrics
        details += f"**Performance Metrics:**\n"
        metrics = strategy_data.get('metrics', {})
        for metric_name, metric_value in sorted(metrics.items()):
            details += f"- {metric_name}: **{metric_value:.4f}**\n"
        
        return details
    
    def set_evaluation_file(self, category: str):
        """Set evaluation file based on category"""
        if category == "all":
            self.evaluation_file = str(Path(EVALUATION_FOLDER) / "../evaluation.json")
        else:
            self.evaluation_file = str(Path(EVALUATION_FOLDER) / f"evaluation_{category}.json")
        
        # Reload queries
        self.load_evaluation_queries()
        
        return f"✓ Loaded {len(self.queries)} queries from category: {category}"
    
    def clear_comparison(self):
        """Clear all comparison results"""
        self.comparison_results = {}
        return "Comparison results cleared!", self.generate_comparison_heatmap(), self.generate_strategy_summary_table(), "Click on a row in the strategy table to see detailed metrics."
    
    def check_data_availability(self):
        """Check how many chunks are available for each chunking strategy"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        strategies = ["fixed", "structure_paragraph", "hierarchical", "parent_child"]
        availability_info = "**Data Availability in Qdrant**\n\n"
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(COLLECTION_NAME)
            total_points = collection_info.points_count
            availability_info += f"Total points in collection: **{total_points}**\n\n"
            
            # Sample a few points to see actual chunking_strategy values
            sample_points = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10,
                with_payload=True
            )[0]
            
            if sample_points:
                unique_strategies = set()
                for point in sample_points:
                    strategy = point.payload.get('chunking_strategy', 'UNKNOWN')
                    unique_strategies.add(strategy)
                
                availability_info += f"**Sample chunking_strategy values found:**\n"
                for s in sorted(unique_strategies):
                    availability_info += f"  - `{s}`\n"
                availability_info += "\n"
            
            availability_info += "**Breakdown by strategy:**\n\n"
            
            for strategy in strategies:
                try:
                    # Count points with this strategy
                    count_result = self.client.count(
                        collection_name=COLLECTION_NAME,
                        count_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="chunking_strategy",
                                    match=MatchValue(value=strategy)
                                )
                            ]
                        )
                    )
                    count = count_result.count
                    
                    if count == 0:
                        availability_info += f"- ❌ **{strategy}**: {count} chunks (NO DATA)\n"
                    elif count < 100:
                        availability_info += f"- ⚠️ **{strategy}**: {count} chunks (LOW)\n"
                    else:
                        availability_info += f"- ✅ **{strategy}**: {count} chunks\n"
                except Exception as e:
                    availability_info += f"- ❌ **{strategy}**: Error - {str(e)}\n"
            
        except Exception as e:
            availability_info += f"\n❌ Error accessing collection: {str(e)}\n"
        
        return availability_info


# Initialize system (will be set when user selects category)
eval_system = EvaluationSystem()


def create_dashboard():
    """Create Gradio dashboard"""
    
    with gr.Blocks(title="RAG Evaluation Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("Evaluation Dashboard")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                
                # Category selector
                category_selector = gr.Dropdown(
                    choices=["doc", "document_law", "fiction", "all"],
                    value="doc",
                    label="📁 Evaluation Category",
                    info="Select which dataset to evaluate"
                )
                
                category_status = gr.Markdown("*Select a category to load evaluation queries*")
                
                chunking_strategy = gr.Dropdown(
                    choices=["fixed", "structure_paragraph", "hierarchical", "parent_child"],
                    value="fixed",
                    label="Chunking Strategy"
                )
                
                retrieval_method = gr.Dropdown(
                    choices=["dense", "sparse", "hybrid", "parent"],
                    value="dense",
                    label="Retrieval Method"
                )
                
                gr.Markdown("### Fair Comparison Settings")
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Top-K Results",
                    info="Number of chunks to retrieve initially"
                )
                
                min_total_tokens = gr.Number(
                    value=2500,
                    label="Min Total Tokens (0 = auto: top_k × 250)",
                    info="For parent_child: enriched context will have more tokens"
                )
                
                max_total_tokens = gr.Number(
                    value=4000,
                    label="Max Total Tokens (0 = unlimited)",
                    info="Stop retrieval when reaching this limit"
                )
                
                with gr.Row():
                    run_btn = gr.Button("🚀 RUN EVALUATION", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear Comparison", variant="secondary", size="lg")
                
                run_all_btn = gr.Button("⚡ RUN ALL COMBINATIONS", variant="primary", size="lg")
                # gr.Markdown("*Runs all chunking strategies with all retrieval methods (parent retrieval only with parent_child)*")
                
                # check_data_btn = gr.Button("🔍 Check Data Availability", variant="secondary")
                
                # gr.Markdown("### System Info")
                # gr.Markdown(f"- **Qdrant:** {QDRANT_HOST}:{QDRANT_PORT}")
                # gr.Markdown(f"- **Collection:** {COLLECTION_NAME}")
                # gr.Markdown(f"- **Model:** {MODEL_NAME}")
                # gr.Markdown(f"- **Queries:** {len(eval_system.queries)}")

                # data_status = gr.Markdown("Click 'Check Data Availability' to see chunk counts")
                clear_status = gr.Textbox(label="Status", visible=False)
            
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                summary_output = gr.Markdown()
                
                with gr.Tab("Metrics Table"):
                    table_output = gr.DataFrame()
                
                with gr.Tab("Detailed Results"):
                    json_output = gr.JSON()
                
                with gr.Tab("📊 Comparison Heatmap"):
                    gr.Markdown("Compare all evaluated strategies side-by-side")
                    comparison_plot = gr.Plot()
                
                with gr.Tab("📋 Strategy Comparison Table"):
                    gr.Markdown("Compare all evaluated strategies")
                    gr.Markdown("💡 **Tip:** Click on any row to see detailed metrics")
                    
                    strategy_table = gr.Dataframe(
                        interactive=False,
                        wrap=True
                    )
        
        # Event handlers
        category_selector.change(
            fn=eval_system.set_evaluation_file,
            inputs=[category_selector],
            outputs=[category_status]
        )
        
        def run_with_params(strategy, method, k, min_tokens, max_tokens):
            # Convert 0 to None for auto calculation
            min_tokens_val = None if min_tokens == 0 else int(min_tokens)
            max_tokens_val = None if max_tokens == 0 else int(max_tokens)
            return eval_system.run_evaluation(strategy, method, int(k), min_tokens_val, max_tokens_val)
        
        run_btn.click(
            fn=run_with_params,
            inputs=[chunking_strategy, retrieval_method, top_k, min_total_tokens, max_total_tokens],
            outputs=[summary_output, table_output, json_output, comparison_plot, strategy_table]
        )
        
        def run_all_with_params(k, min_tokens, max_tokens):
            # Convert 0 to None for auto calculation
            min_tokens_val = None if min_tokens == 0 else int(min_tokens)
            max_tokens_val = None if max_tokens == 0 else int(max_tokens)
            return eval_system.run_all_evaluations(int(k), min_tokens_val, max_tokens_val)
        
        run_all_btn.click(
            fn=run_all_with_params,
            inputs=[top_k, min_total_tokens, max_total_tokens],
            outputs=[summary_output, table_output, json_output, comparison_plot, strategy_table]
        )
        
        clear_btn.click(
            fn=eval_system.clear_comparison,
            inputs=[],
            outputs=[clear_status, comparison_plot, strategy_table]
        )
        
        # check_data_btn.click(
        #     fn=eval_system.check_data_availability,
        #     inputs=[],
        #     outputs=[data_status]
        # )
    
    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(share=False, server_port=7860)
