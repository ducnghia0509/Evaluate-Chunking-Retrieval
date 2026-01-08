import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys

# Import dense and sparse retrievers
sys.path.append(str(Path(__file__).parent))
from dense_retrieval import DenseRetrieval
from sparse_retrieval import SparseRetrieval


class HybridRetrieval:
    """Hybrid retrieval combining dense and sparse methods"""
    
    def __init__(self, config_path: str = "../../Retrieval/config.json", collection_name: str = "evaluate"):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.common_config = config['common']
        self.retrieval_config = config['hybrid_retrieval']
        
        self.top_k = self.common_config.get('top_k', 10)
        self.dense_weight = self.retrieval_config.get('dense_weight', 0.7)
        self.sparse_weight = self.retrieval_config.get('sparse_weight', 0.3)
        self.strategy_type = self.retrieval_config.get('strategy', 'weighted_sum')
        
        # Initialize retrievers with shared collection
        self.dense_retriever = DenseRetrieval(config_path, collection_name)
        self.sparse_retriever = SparseRetrieval(config_path, collection_name)
    
    def build_index(self, embedd_dir: str, strategy: str = "fixed"):
        """Build both dense and sparse indices"""
        print(f"\nBuilding hybrid index for strategy: {strategy}")
        print("="*60)
        
        # Build dense index
        print("\n[1/2] Building dense (vector) index...")
        self.dense_retriever.build_index(embedd_dir, strategy)
        
        # Build sparse index
        print("\n[2/2] Building sparse (BM25) index...")
        self.sparse_retriever.build_index(embedd_dir, strategy)
        
        print("\n" + "="*60)
        print("Hybrid index built successfully!")
    
    def normalize_scores(self, scores: List[float], method: str = 'z_score') -> List[float]:
        if not scores or len(scores) == 0:
            return []
        
        scores = np.array(scores)
        
        if method == 'min_max':
            min_score = scores.min()
            max_score = scores.max()
            range_val = max_score - min_score
            
            # Xử lý trường hợp tất cả scores bằng nhau
            if range_val == 0:
                # Trả về mảng toàn 0.5 hoặc 1.0
                return np.ones_like(scores).tolist()  # hoặc np.full_like(scores, 0.5)
            
            return ((scores - min_score) / range_val).tolist()
        
        elif method == 'z_score':
            mean = scores.mean()
            std = scores.std()
            
            # Avoid division by zero
            if std == 0:
                return np.ones_like(scores).tolist()
            
            # Normalize to approximate normal distribution
            normalized = (scores - mean) / std
            
            # Optional: transform to [0, 1] range with sigmoid
            # sigmoid = 1 / (1 + np.exp(-normalized))
            return normalized.tolist()
        
        else:
            # Default to z_score
            return self.normalize_scores(scores, method='z_score')
        
    def rrf_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60):
        """Reciprocal Rank Fusion implementation"""
        rrf_scores = {}
        
        # Tạo rank dictionaries
        dense_ranks = {r['chunk_id']: i+1 for i, r in enumerate(dense_results)}
        sparse_ranks = {r['chunk_id']: i+1 for i, r in enumerate(sparse_results)}
        
        # Tính RRF scores
        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        for chunk_id in all_chunk_ids:
            dense_rank = dense_ranks.get(chunk_id, len(dense_results) + 1)
            sparse_rank = sparse_ranks.get(chunk_id, len(sparse_results) + 1)
            
            rrf_score = (1.0 / (dense_rank + k)) + (1.0 / (sparse_rank + k))
            rrf_scores[chunk_id] = rrf_score
        
        return rrf_scores

    def merge_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """
        Merge and re-rank results from dense and sparse retrievers
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
        
        Returns:
            Merged and re-ranked results
        """
        # Create score dictionaries
        dense_scores = {r['chunk_id']: r['retrieval_score'] for r in dense_results}
        sparse_scores = {r['chunk_id']: r['retrieval_score'] for r in sparse_results}
        
        # Normalize scores
        if dense_scores:
            dense_scores_list = list(dense_scores.values())
            normalized_dense = self.normalize_scores(dense_scores_list)
            dense_scores = {cid: score for cid, score in zip(dense_scores.keys(), normalized_dense)}
        
        if sparse_scores:
            sparse_scores_list = list(sparse_scores.values())
            normalized_sparse = self.normalize_scores(sparse_scores_list)
            sparse_scores = {cid: score for cid, score in zip(sparse_scores.keys(), normalized_sparse)}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Create chunk lookup
        chunk_lookup = {}
        for r in dense_results + sparse_results:
            chunk_lookup[r['chunk_id']] = r
        
        # Calculate hybrid scores
        hybrid_results = []
        if self.strategy_type == 'rrf':
            rrf_scores = self.rrf_fusion(dense_results, sparse_results)

        for chunk_id in all_chunk_ids:
            dense_score = dense_scores.get(chunk_id, 0.0)
            sparse_score = sparse_scores.get(chunk_id, 0.0)

            if self.strategy_type == 'weighted_sum':
                hybrid_score = (
                    self.dense_weight * dense_score +
                    self.sparse_weight * sparse_score
                )

            elif self.strategy_type == 'max':
                hybrid_score = max(dense_score, sparse_score)

            elif self.strategy_type == 'rrf':
                hybrid_score = rrf_scores.get(chunk_id, 0.0)

            else:
                hybrid_score = (
                    self.dense_weight * dense_score +
                    self.sparse_weight * sparse_score
                )

            chunk = chunk_lookup[chunk_id].copy()
            chunk.update({
                'retrieval_score': float(hybrid_score),
                'retrieval_method': 'hybrid',
                'dense_score': float(dense_score),
                'sparse_score': float(sparse_score),
            })

            hybrid_results.append(chunk)


        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        
        return hybrid_results[:self.top_k]
    
    def search(self, query: str, top_k: int = None, chunking_strategy: str = None, metadata_filter: Dict = None) -> List[Dict]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunking_strategy: Chunking strategy to filter
            metadata_filter: Optional filter on metadata fields
                Examples:
                - {'metadata.strategy': 'hierarchical'}
                - {'metadata.section_header': 'Introduction'}
                - {'source_file': 'example.md'}
        
        Returns:
            List of retrieved chunks with hybrid scores
        """
        k = top_k or self.top_k
        
        # Get results from both retrievers (retrieve more for merging)
        retrieve_k = k * 2  # Retrieve more candidates for better merging
        
        # Dense retrieval with chunking_strategy filter
        dense_results = self.dense_retriever.search(query, retrieve_k, chunking_strategy, metadata_filter)
        
        # Sparse retrieval with chunking_strategy filter
        sparse_results = self.sparse_retriever.search(query, retrieve_k, chunking_strategy)
        
        # Apply metadata filter to sparse results if needed
        if metadata_filter:
            sparse_results = self._filter_results(sparse_results, metadata_filter)
        
        # Merge and re-rank
        hybrid_results = self.merge_results(dense_results, sparse_results)
        
        return hybrid_results[:k]
    
    def _filter_results(self, results: List[Dict], metadata_filter: Dict) -> List[Dict]:
        """
        Manually filter results based on metadata
        
        Args:
            results: List of results to filter
            metadata_filter: Filter conditions
        
        Returns:
            Filtered results
        """
        if not metadata_filter:
            return results
        
        filtered = []
        for result in results:
            match = True
            for key, value in metadata_filter.items():
                # Handle nested metadata access (e.g., 'metadata.strategy')
                if '.' in key:
                    parts = key.split('.')
                    obj = result
                    try:
                        for part in parts:
                            obj = obj[part]
                        if obj != value:
                            match = False
                            break
                    except (KeyError, TypeError):
                        match = False
                        break
                else:
                    # Direct key access
                    if result.get(key) != value:
                        match = False
                        break
            
            if match:
                filtered.append(result)
        
        return filtered
    
    def batch_search(self, queries: List[str], top_k: int = None, metadata_filter: Dict = None) -> List[List[Dict]]:
        """Batch search for multiple queries"""
        return [self.search(q, top_k, metadata_filter) for q in queries]


def process_retrieval(embedd_dir: str, strategy: str, queries_file: str = None,
                     config_path: str = "../../Retrieval/config.json"):
    """
    Process hybrid retrieval for a chunking strategy
    
    Args:
        embedd_dir: Path to Embedd folder
        strategy: Chunking strategy name
        queries_file: Optional path to queries JSON file
        config_path: Path to config file
    """
    # Initialize retriever
    retriever = HybridRetrieval(config_path)
    
    # Build indices
    retriever.build_index(embedd_dir, strategy)
    
    # If queries provided, run retrieval
    if queries_file and Path(queries_file).exists():
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        queries = queries_data.get('queries', [])
        
        print(f"\nRunning hybrid retrieval for {len(queries)} queries...")
        
        all_results = []
        for query_item in queries:
            query_text = query_item.get('query', query_item)
            results = retriever.search(query_text)
            
            all_results.append({
                'query': query_text,
                'results': results
            })
        
        # Save results
        output_dir = Path("../../Retrieval/results/hybrid") / strategy
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
