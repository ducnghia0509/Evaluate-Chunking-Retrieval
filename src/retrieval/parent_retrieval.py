import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Distance, VectorParams
from sentence_transformers import SentenceTransformer
import torch
import tiktoken


class ParentRetrieval:
    """
    Parent-Child retrieval using single collection 'evaluate'
    """
    
    def __init__(self, config_path: str = "../../Retrieval/config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.common_config = config['common']
        self.retrieval_config = config.get('parent_retrieval', {})
        
        self.top_k = self.common_config.get('top_k', 10)
        self.max_parents = self.retrieval_config.get('max_parents', 10)  # Increased from 5
        self.aggregation = self.retrieval_config.get('aggregation', 'max_score')
        self.context_window = self.retrieval_config.get('focused_parent_window', 500)
        # Option to return full parent instead of sliced
        self.return_full_parent = self.retrieval_config.get('return_full_parent', True)
        
        # Collection name - DÙNG CHUNG
        self.collection_name = "evaluate"
        
        # Client từ config hoặc default
        qdrant_config = config.get('qdrant', {})
        self.client = QdrantClient(
            host=qdrant_config.get('host', 'localhost'),
            port=qdrant_config.get('port', 6333)
        )
        
        # Embedding model từ dense config
        dense_config = config.get('dense_retrieval', {})
        model_name = dense_config.get('embedding_model', 'Alibaba-NLP/gte-multilingual-base')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            # Fallback model
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Kiểm tra collection
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"✓ ParentRetrieval connected to collection '{self.collection_name}'")
            print(f"  Collection has {collection_info.points_count} points")
        except Exception as e:
            print(f"⚠️ Warning: Could not access collection: {e}")
    
    def _slice_parent(self, parent_text: str, start_token: int, end_token: int) -> str:
        """Slice parent text around child match"""
        try:
            tokens = self.tokenizer.encode(parent_text)
            center = (start_token + end_token) // 2
            half = self.context_window // 2
            
            left = max(0, center - half)
            right = min(len(tokens), center + half)
            
            return self.tokenizer.decode(tokens[left:right])
        except:
            # Fallback: simple substring
            return parent_text
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Parent retrieval từ collection chung 'evaluate'
        
        Steps:
        1. Tìm child chunks (metadata.role = 'child')
        2. Aggregate child scores by parent_id
        3. Tìm parent text
        4. Slice parent around best child
        """
        k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        # Tìm child chunks
        child_filter = Filter(
            must=[
                FieldCondition(
                    key="chunking_strategy",
                    match=MatchValue(value="parent_child")
                ),
                FieldCondition(
                    key="metadata.role",
                    match=MatchValue(value="child")
                )
            ]
        )
        
        try:
            # Retrieve more children to get better parent coverage
            # Increased from k*3 to k*5 for better recall
            child_hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=child_filter,
                limit=k * 5,  # Increased multiplier
                with_payload=True
            ).points
        except Exception as e:
            print(f"Error querying child chunks: {e}")
            return []
        
        # Aggregate child scores by parent_id
        parent_scores = {}
        child_matches = {}
        
        for hit in child_hits:
            metadata = hit.payload.get('metadata', {})
            parent_id = metadata.get('parent_id')
            
            if not parent_id:
                continue
            
            parent_scores.setdefault(parent_id, []).append(float(hit.score))
            child_matches.setdefault(parent_id, []).append({
                "chunk_id": hit.payload.get('chunk_id'),
                "start_token": metadata.get('start_token', 0),
                "end_token": metadata.get('end_token', 0)
            })
        
        # Lấy parent chunks
        if not parent_scores:
            return []
        
        # Tìm parent text cho các parent_id có score
        # Fetch all parent chunks at once
        parents = {}
        
        for parent_id in parent_scores.keys():
            parent_filter = Filter(
                must=[
                    FieldCondition(
                        key="chunking_strategy",
                        match=MatchValue(value="parent_child")
                    ),
                    FieldCondition(
                        key="metadata.role",
                        match=MatchValue(value="parent")
                    ),
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=parent_id)
                    )
                ]
            )
            
            try:
                parent_results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=parent_filter,
                    limit=1,
                    with_payload=True
                )[0]
                
                if parent_results:
                    parents[parent_id] = parent_results[0].payload.get('text', '')
            except Exception as e:
                print(f"Warning: Could not fetch parent {parent_id}: {e}")
                continue
        
        # Tạo kết quả
        results = []
        
        for pid, scores in list(parent_scores.items())[:self.max_parents]:
            if pid not in parents:
                continue
            
            children = child_matches.get(pid, [])
            if not children:
                continue
            
            # Tìm child có độ dài lớn nhất
            best_child = max(
                children,
                key=lambda x: x.get("end_token", 0) - x.get("start_token", 0)
            )
            
            # Return full parent text or sliced window based on config
            if self.return_full_parent:
                # Use full parent text for better context
                focused_text = parents[pid]
            else:
                # Slice parent text around best child
                focused_text = self._slice_parent(
                    parents[pid],
                    best_child.get("start_token", 0),
                    best_child.get("end_token", 0)
                )
            
            results.append({
                "parent_id": pid,
                "retrieval_score": max(scores),
                "context": focused_text,
                "num_child_hits": len(scores),
                "retrieval_method": "parent",
                "full_parent": self.return_full_parent
            })
        
        results.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return results[:k]