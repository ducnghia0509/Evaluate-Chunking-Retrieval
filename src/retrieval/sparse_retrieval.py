import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import re


class SparseRetrieval:
    """Sparse retrieval using BM25"""
    
    def __init__(self, config_path: str = "../../Retrieval/config.json", collection_name: str = "evaluate"):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.common_config = config['common']
        self.retrieval_config = config['sparse_retrieval']
        
        self.top_k = self.common_config.get('top_k', 10)
        self.k1 = self.retrieval_config.get('k1', 1.5)
        self.b = self.retrieval_config.get('b', 0.75)
        
        # Initialize Qdrant client to fetch chunks
        self.client = QdrantClient(
            host='localhost',
            port=6333
        )
        self.collection_name = collection_name
        
        self.bm25_indices = {}  # Strategy -> BM25 index
        self.chunks_by_strategy = {}  # Strategy -> chunks
        self.tokenized_corpus_by_strategy = {}  # Strategy -> tokenized corpus
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be improved with proper tokenizer)"""
        # Lowercase and split by word boundaries
        text = text.lower()
        # Keep Vietnamese and alphanumeric characters
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def build_index_for_strategy(self, strategy: str):
        """
        Build BM25 index from Qdrant for specific strategy
        
        Args:
            strategy: Chunking strategy name
        """
        if strategy in self.bm25_indices:
            print(f"BM25 index for {strategy} already built")
            return
        
        print(f"\nBuilding BM25 index for strategy: {strategy}")
        
        # Fetch all chunks for this strategy from Qdrant
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="chunking_strategy",
                        match=MatchValue(value=strategy)
                    )
                ]
            )
            
            all_chunks = []
            offset = None
            
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=search_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
                
                points, next_offset = result
                
                if not points:
                    break
                
                for point in points:
                    all_chunks.append({
                        'chunk_id': point.payload.get('chunk_id', ''),
                        'text': point.payload.get('text', ''),
                        'source_file': point.payload.get('source_file', ''),
                        'metadata': point.payload.get('metadata', {})
                    })
                
                if next_offset is None:
                    break
                offset = next_offset
            
            if len(all_chunks) == 0:
                print(f"⚠️ No chunks found for strategy: {strategy}")
                return
            
            # Tokenize all chunks
            tokenized_corpus = [self.tokenize(chunk['text']) for chunk in all_chunks]
            
            # Build BM25 index
            bm25_index = BM25Okapi(
                tokenized_corpus,
                k1=self.k1,
                b=self.b
            )
            
            # Store
            self.bm25_indices[strategy] = bm25_index
            self.chunks_by_strategy[strategy] = all_chunks
            self.tokenized_corpus_by_strategy[strategy] = tokenized_corpus
            
            print(f"✓ BM25 index built with {len(all_chunks)} chunks")
            print(f"  Parameters: k1={self.k1}, b={self.b}")
            
        except Exception as e:
            print(f"⚠️ Error building BM25 index for {strategy}: {e}")
    
    def search(self, query: str, top_k: int = None, chunking_strategy: str = None) -> List[Dict]:
        """
        Search for relevant chunks using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunking_strategy: Chunking strategy to search in
        
        Returns:
            List of retrieved chunks with BM25 scores
        """
        if not chunking_strategy:
            raise ValueError("chunking_strategy is required")
        
        # Build index for this strategy if not exists
        if chunking_strategy not in self.bm25_indices:
            self.build_index_for_strategy(chunking_strategy)
        
        if chunking_strategy not in self.bm25_indices:
            print(f"⚠️ No BM25 index for strategy: {chunking_strategy}")
            return []
        
        bm25 = self.bm25_indices[chunking_strategy]
        chunks = self.chunks_by_strategy[chunking_strategy]
        
        k = top_k or self.top_k
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            chunk = chunks[idx].copy()
            chunk['retrieval_score'] = float(scores[idx])
            chunk['retrieval_method'] = 'bm25'
            results.append(chunk)
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[Dict]]:
        """Batch search for multiple queries"""
        return [self.search(q, top_k) for q in queries]


def process_retrieval(embedd_dir: str, strategy: str, queries_file: str = None,
                     config_path: str = "../../Retrieval/config.json"):
    """
    Process sparse (BM25) retrieval for a chunking strategy
    
    Args:
        embedd_dir: Path to Embedd folder
        strategy: Chunking strategy name
        queries_file: Optional path to queries JSON file
        config_path: Path to config file
    """
    # Initialize retriever
    retriever = SparseRetrieval(config_path)
    
    # Build index
    retriever.build_index(embedd_dir, strategy)
    
    # If queries provided, run retrieval
    if queries_file and Path(queries_file).exists():
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        queries = queries_data.get('queries', [])
        
        print(f"\nRunning BM25 retrieval for {len(queries)} queries...")
        
        all_results = []
        for query_item in queries:
            query_text = query_item.get('query', query_item)
            results = retriever.search(query_text)
            
            all_results.append({
                'query': query_text,
                'results': results
            })
        
        # Save results
        output_dir = Path("../../Retrieval/results/sparse") / strategy
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "retrieval_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")
    
    return retriever


if __name__ == "__main__":
    # Example usage
    embedd_dir = "../../Embedd"
    strategy = "fixed"
    
    retriever = process_retrieval(embedd_dir, strategy)
