# file: evaluation_metrics.py
"""
Evaluation metrics for retrieval systems
"""

from typing import List, Dict, Any, Set
import numpy as np
from collections import defaultdict


class RetrievalMetrics:
    """Calculate various retrieval metrics"""
    
    @staticmethod
    def recall_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
        """Recall@k: Proportion of relevant items retrieved in top-k"""
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len([id_ for id_ in retrieved_at_k if id_ in relevant_ids])
        return relevant_retrieved / len(relevant_ids)
    
    @staticmethod
    def precision_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
        """Precision@k: Proportion of retrieved items that are relevant in top-k"""
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len([id_ for id_ in retrieved_at_k if id_ in relevant_ids])
        return relevant_retrieved / k
    
    @staticmethod
    def average_precision(relevant_ids: Set[str], retrieved_ids: List[str]) -> float:
        """Average Precision (AP)"""
        if not relevant_ids:
            return 0.0
        
        relevant_retrieved = 0
        sum_precision = 0.0
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                relevant_retrieved += 1
                sum_precision += relevant_retrieved / i
        
        return sum_precision / len(relevant_ids) if relevant_ids else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(relevant_ids: Set[str], retrieved_ids: List[str]) -> float:
        """Mean Reciprocal Rank (MRR)"""
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(relevant_ids: Set[str], retrieved_ids: List[str], scores: List[float], k: int) -> float:
        """Normalized Discounted Cumulative Gain (nDCG@k)"""
        if not relevant_ids:
            return 0.0
        
        # DCG@k
        dcg = 0.0
        for i, (doc_id, score) in enumerate(zip(retrieved_ids[:k], scores[:k]), 1):
            if doc_id in relevant_ids:
                dcg += score / np.log2(i + 1)  # log2(i+1) for discount
        
        # Ideal DCG@k (sort relevant items by score)
        ideal_scores = sorted([s for d, s in zip(retrieved_ids, scores) if d in relevant_ids], reverse=True)
        idcg = sum(score / np.log2(i + 1) for i, score in enumerate(ideal_scores[:k], 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_all_metrics(
        relevant_ids: Set[str],
        retrieved_ids: List[str],
        scores: List[float],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Calculate all metrics at different k values"""
        metrics = {}
        
        # Recall@k
        for k in k_values:
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(relevant_ids, retrieved_ids, k)
        
        # Precision@k
        for k in k_values:
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(relevant_ids, retrieved_ids, k)
        
        # MRR
        metrics["mrr"] = RetrievalMetrics.mean_reciprocal_rank(relevant_ids, retrieved_ids)
        
        # nDCG@k
        for k in k_values:
            metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(relevant_ids, retrieved_ids, scores, k)
        
        # Mean Average Precision (MAP) - AP for single query
        metrics["map"] = RetrievalMetrics.average_precision(relevant_ids, retrieved_ids)
        
        return metrics
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across multiple queries"""
        aggregated = defaultdict(list)
        
        # Collect all metrics
        for metrics in metrics_list:
            for key, value in metrics.items():
                aggregated[key].append(value)
        
        # Calculate mean and std
        result = {}
        for key, values in aggregated.items():
            result[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        return result


def format_metrics_report(metrics: Dict[str, float], indent: str = "  ") -> str:
    """Format metrics for readable output"""
    lines = []
    
    # Group by metric type
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith('recall')}
    precision_metrics = {k: v for k, v in metrics.items() if k.startswith('precision')}
    ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg')}
    other_metrics = {k: v for k, v in metrics.items() if k in ['mrr', 'map']}
    
    lines.append("Recall Metrics:")
    for k in sorted(recall_metrics.keys()):
        lines.append(f"{indent}{k}: {recall_metrics[k]:.4f}")
    
    lines.append("\nPrecision Metrics:")
    for k in sorted(precision_metrics.keys()):
        lines.append(f"{indent}{k}: {precision_metrics[k]:.4f}")
    
    lines.append("\nNDCG Metrics:")
    for k in sorted(ndcg_metrics.keys()):
        lines.append(f"{indent}{k}: {ndcg_metrics[k]:.4f}")
    
    lines.append("\nOther Metrics:")
    for k, v in other_metrics.items():
        lines.append(f"{indent}{k}: {v:.4f}")
    
    return "\n".join(lines)