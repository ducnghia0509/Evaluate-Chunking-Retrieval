import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import torch
from uuid import uuid4


class DenseRetrieval:
    """Dense retrieval using Qdrant vector database"""
    
    def __init__(self, config_path: str = "../../Retrieval/config.json", collection_name: str = "evaluate"):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.common_config = config['common']
        self.retrieval_config = config['dense_retrieval']
        self.qdrant_config = config['qdrant']
        
        self.top_k = self.common_config.get('top_k', 10)
        self.embedding_dim = self.retrieval_config.get('embedding_dim', 768)
        self.metric = self.retrieval_config.get('metric', 'cosine')
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.qdrant_config.get('host', 'localhost'),
            port=self.qdrant_config.get('port', 6333)
        )
        
        # Load embedding model
        model_name = self.retrieval_config.get('embedding_model')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
        # Use shared collection
        self.collection_name = collection_name
        print(f"Using collection: {self.collection_name}")
    
    def build_index(self, embedd_dir: str, strategy: str = "fixed"):
        """
        Build Qdrant collection from embedded chunks
        
        Args:
            embedd_dir: Path to Embedd folder
            strategy: Chunking strategy name (fixed, structure_paragraph, etc.)
        """
        print(f"\nBuilding Qdrant collection for strategy: {strategy}")
        
        embedd_path = Path(embedd_dir) / strategy
        
        if not embedd_path.exists():
            raise ValueError(f"Embedd path not found: {embedd_path}")
        
        # Create collection name
        self.collection_name = f"{self.qdrant_config.get('collection_prefix', 'chunks')}_{strategy}"
        
        # Delete collection if exists
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except:
            pass
        
        # Create collection
        distance_map = {
            'cosine': Distance.COSINE,
            'euclidean': Distance.EUCLID,
            'dot': Distance.DOT
        }
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=distance_map.get(self.metric.lower(), Distance.COSINE),
                on_disk=self.qdrant_config.get('on_disk', False)
            )
        )
        
        # Load all chunks
        points = []
        point_id = 0
        
        for json_file in embedd_path.rglob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            if not isinstance(chunks_data, list):
                continue
            
            for chunk in chunks_data:
                if 'embedding' not in chunk:
                    print(f"Warning: No embedding found in {json_file.name}")
                    continue
                
                # Prepare payload (metadata)
                payload = {
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'source_file': chunk.get('source_file', ''),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'metadata': chunk.get('metadata', {})
                }
                
                # Create point
                points.append(PointStruct(
                    id=point_id,
                    vector=chunk['embedding'],
                    payload=payload
                ))
                point_id += 1
                
                # Batch upload every 100 points
                if len(points) >= 100:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    points = []
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        # Get collection info
        info = self.client.get_collection(self.collection_name)
        print(f"Collection created: {self.collection_name}")
        print(f"Total points: {info.points_count}")
        print(f"Vector size: {self.embedding_dim}")
        print(f"Distance metric: {self.metric}")
    
    def search(self, query: str, top_k: int = None, chunking_strategy: str = None, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for relevant chunks given a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunking_strategy: Chunking strategy to filter (e.g., 'fixed', 'parent_child')
            metadata_filter: Optional filter on metadata (e.g., {'source_file': 'doc1.md'})
        
        Returns:
            List of retrieved chunks with scores
        """
        if self.collection_name is None:
            raise ValueError("Collection not set. Initialize with collection_name.")
        
        k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=(self.metric == 'cosine')
        ).tolist()
        
        # Build filter
        conditions = []
        
        # Add chunking_strategy filter
        if chunking_strategy:
            conditions.append(FieldCondition(
                key="chunking_strategy",
                match=MatchValue(value=chunking_strategy)
            ))
        
        # Add metadata filters
        if metadata_filter:
            for key, value in metadata_filter.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))
        
        query_filter = Filter(must=conditions) if conditions else None
        
        # Search in Qdrant
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            query_filter=query_filter,
            with_payload=True
        ).points
        
        # Prepare results
        results = []
        for hit in search_results:
            result = {
                'chunk_id': hit.payload['chunk_id'],
                'text': hit.payload['text'],
                'source_file': hit.payload.get('source_file', ''),
                'chunk_index': hit.payload.get('chunk_index', 0),
                'metadata': hit.payload.get('metadata', {}),
                'retrieval_score': float(hit.score),
                'retrieval_method': 'dense'
            }
            results.append(result)
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = None, metadata_filter: Optional[Dict] = None) -> List[List[Dict]]:
        """Batch search for multiple queries"""
        return [self.search(q, top_k, metadata_filter) for q in queries]
    
    def delete_collection(self):
        """Delete the Qdrant collection"""
        if self.collection_name:
            self.client.delete_collection(self.collection_name)
            print(f"Collection deleted: {self.collection_name}")


def process_retrieval(embedd_dir: str, strategy: str, queries_file: str = None, 
                     config_path: str = "../../Retrieval/config.json"):
    """
    Process dense retrieval for a chunking strategy
    
    Args:
        embedd_dir: Path to Embedd folder
        strategy: Chunking strategy name
        queries_file: Optional path to queries JSON file
        config_path: Path to config file
    """
    # Initialize retriever
    retriever = DenseRetrieval(config_path)
    
    # Build index
    retriever.build_index(embedd_dir, strategy)
    
    # If queries provided, run retrieval
    if queries_file and Path(queries_file).exists():
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        queries = queries_data.get('queries', [])
        
        print(f"\nRunning retrieval for {len(queries)} queries...")
        
        all_results = []
        for query_item in queries:
            query_text = query_item.get('query', query_item)
            results = retriever.search(query_text)
            
            all_results.append({
                'query': query_text,
                'results': results
            })
        
        # Save results
        output_dir = Path("../../Retrieval/results/dense") / strategy
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
