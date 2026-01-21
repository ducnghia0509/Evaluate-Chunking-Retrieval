"""
Sử dụng colbert để biểu diễn query và các candidate thành nhiều vector thấp chiều, so sánh để ranking lại
"""

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class MultivectorRetrieval:
    """
    Multivector Retrieval using ColBERT for late interaction
    - First stage: Dense retrieval to get candidate chunks
    - Second stage: ColBERT reranking for fine-grained matching
    """
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "evaluate",
                 colbert_model: str = "colbert-ir/colbertv2.0",
                 rerank_top_k: int = 20):
        """
        Initialize Multivector Retrieval
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Collection name in Qdrant
            colbert_model: ColBERT model name
            rerank_top_k: Number of candidates to retrieve before reranking
        """
        # Initialize Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Initialize ColBERT model for reranking
        print(f"Loading ColBERT model: {colbert_model}...")
        self.colbert_model = LateInteractionTextEmbedding(colbert_model)
        print("✓ ColBERT model loaded")
        
        # Reranking parameters
        self.rerank_top_k = rerank_top_k
    
    def _dense_retrieval(self, query: str, chunking_strategy: str, top_k: int) -> List[Dict]:
        """
        First stage: Dense retrieval to get candidates
        
        Args:
            query: Search query
            chunking_strategy: Chunking strategy filter
            top_k: Number of candidates to retrieve
        
        Returns:
            List of candidate chunks
        """
        # Use standard dense retrieval from Qdrant
        # This assumes embeddings are already in Qdrant
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Load embedding model (same as used for indexing)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(
            "Alibaba-NLP/gte-multilingual-base", 
            device=device,
            trust_remote_code=True
        )
        
        # Encode query
        query_embedding = embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        # Create filter
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="chunking_strategy",
                    match=MatchValue(value=chunking_strategy)
                )
            ]
        )
        
        # For parent_child, only retrieve child chunks
        if chunking_strategy == "parent_child":
            search_filter.must.append(
                FieldCondition(
                    key="metadata.role",
                    match=MatchValue(value="child")
                )
            )
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True
        ).points
        
        # Convert to dict
        chunks = []
        for hit in results:
            chunk_data = {
                'chunk_id': hit.payload['chunk_id'],
                'text': hit.payload['text'],
                'dense_score': float(hit.score),  # Store dense score
                'source_file': hit.payload.get('source_file', ''),
                'category': hit.payload.get('category', ''),
                'metadata': hit.payload.get('metadata', {})
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _colbert_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Second stage: ColBERT reranking
        
        Args:
            query: Search query
            candidates: Candidate chunks from dense retrieval
            top_k: Number of results to return after reranking
        
        Returns:
            Reranked chunks with ColBERT scores
        """
        if not candidates:
            return []
        
        # Extract texts
        candidate_texts = [c['text'] for c in candidates]
        
        # Embed query and candidates with ColBERT
        # ColBERT produces token-level embeddings
        query_embeddings = list(self.colbert_model.query_embed([query]))[0]
        passage_embeddings = list(self.colbert_model.passage_embed(candidate_texts))
        
        # Calculate late interaction scores
        # MaxSim: for each query token, find max similarity with all passage tokens
        scores = []
        for i, passage_emb in enumerate(passage_embeddings):
            # Calculate similarity matrix between query and passage tokens
            # query_emb: (num_query_tokens, dim)
            # passage_emb: (num_passage_tokens, dim)
            
            # Calculate cosine similarity
            similarity_matrix = np.dot(query_embeddings, passage_emb.T)
            
            # MaxSim: for each query token, take max similarity with passage
            max_sims = np.max(similarity_matrix, axis=1)
            
            # Average over query tokens
            score = np.mean(max_sims)
            scores.append(score)
        
        # Sort by ColBERT score
        scored_candidates = [
            {**candidate, 'colbert_score': float(score), 'score': float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        scored_candidates.sort(key=lambda x: x['colbert_score'], reverse=True)
        
        return scored_candidates[:top_k]
    
    def search(self, query: str, chunking_strategy: str, top_k: int = 10, 
               use_reranking: bool = True) -> List[Dict]:
        """
        Search with optional ColBERT reranking
        
        Args:
            query: Search query
            chunking_strategy: Chunking strategy filter
            top_k: Number of final results
            use_reranking: Whether to use ColBERT reranking
        
        Returns:
            List of retrieved chunks (with or without reranking)
        """
        if use_reranking:
            # Two-stage retrieval
            print(f"🔍 Stage 1: Dense retrieval (top-{self.rerank_top_k} candidates)")
            candidates = self._dense_retrieval(
                query, 
                chunking_strategy, 
                top_k=self.rerank_top_k
            )
            
            if not candidates:
                print("⚠️ No candidates found in dense retrieval")
                return []
            
            print(f"🔍 Stage 2: ColBERT reranking (top-{top_k} from {len(candidates)} candidates)")
            reranked = self._colbert_rerank(query, candidates, top_k)
            
            print(f"✓ Retrieved {len(reranked)} chunks with ColBERT reranking")
            return reranked
        else:
            # Dense retrieval only
            print(f"🔍 Dense retrieval only (top-{top_k})")
            results = self._dense_retrieval(query, chunking_strategy, top_k)
            
            # Rename dense_score to score for consistency
            for r in results:
                r['score'] = r.get('dense_score', 0.0)
            
            print(f"✓ Retrieved {len(results)} chunks")
            return results


# Test function
if __name__ == "__main__":
    # Initialize retrieval
    retriever = MultivectorRetrieval()
    
    # Test query
    query = "Nguyên tắc AI của Google là gì?"
    
    # Test with reranking
    print("\n" + "="*80)
    print("TEST: With ColBERT Reranking")
    print("="*80)
    results_with_rerank = retriever.search(
        query=query,
        chunking_strategy="fixed",
        top_k=5,
        use_reranking=True
    )
    
    print("\nTop 3 results:")
    for i, result in enumerate(results_with_rerank[:3], 1):
        print(f"\n{i}. Score: {result['colbert_score']:.4f}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Text: {result['text'][:100]}...")
    
    # Test without reranking
    print("\n" + "="*80)
    print("TEST: Without ColBERT Reranking (Dense only)")
    print("="*80)
    results_without_rerank = retriever.search(
        query=query,
        chunking_strategy="fixed",
        top_k=5,
        use_reranking=False
    )
    
    print("\nTop 3 results:")
    for i, result in enumerate(results_without_rerank[:3], 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Text: {result['text'][:100]}...")