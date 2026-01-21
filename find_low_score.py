# từ folder Evaluate, duyệt qua mọi file,  thống kê lại xem những câu nào (top-10) nhiều nhất, có recall thấp nhất

import json
import os
from pathlib import Path
from collections import defaultdict

def find_low_score_queries():
    # Path to Evaluate/results folder
    results_folder = Path(__file__).parent / "Evaluate" / "results"
    
    # Store all query results with their recall scores, grouped by query_id
    query_data = defaultdict(list)
    
    # Iterate through all JSON files
    for json_file in results_folder.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Process each result in the file
            for result in data.get("results", []):
                query_id = result.get("query_id")
                question = result.get("question", "")
                metrics = result.get("metrics", {})
                recall_10 = metrics.get("recall@10", 0)
                recall_5 = metrics.get("recall@5", 0)
                recall_3 = metrics.get("recall@3", 0)
                recall_1 = metrics.get("recall@1", 0)
                
                chunking_strategy = result.get("chunking_strategy")
                retrieval_method = result.get("retrieval_method")
                timestamp = result.get("timestamp", "")
                
                query_data[query_id].append({
                    "query_id": query_id,
                    "question": question,
                    "recall@1": recall_1,
                    "recall@3": recall_3,
                    "recall@5": recall_5,
                    "recall@10": recall_10,
                    "file": json_file.name,
                    "chunking": chunking_strategy,
                    "retrieval": retrieval_method,
                    "timestamp": timestamp
                })
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    # Keep only the latest result for each query_id (based on timestamp)
    query_recalls = []
    for query_id, results in query_data.items():
        # Sort by timestamp and take the latest
        latest = max(results, key=lambda x: x["timestamp"])
        query_recalls.append(latest)
    
    print(f"\nTotal unique queries: {len(query_recalls)}")
    
    # Function to print top 10 for a specific recall metric
    def print_top_10(metric_name, query_list):
        sorted_queries = sorted(query_list, key=lambda x: (x[metric_name], x["recall@10"], x["recall@5"], x["recall@3"], x["recall@1"]))
        
        print("\n" + "="*100)
        print(f"TOP 10 QUERIES WITH LOWEST {metric_name.upper()}")
        print("="*100)
        
        for i, result in enumerate(sorted_queries[:10], 1):
            print(f"\n{i}. Query ID: {result['query_id']}")
            print(f"   File: {result['file']}")
            print(f"   Chunking: {result['chunking']} | Retrieval: {result['retrieval']}")
            print(f"   Recall@1: {result['recall@1']:.3f} | Recall@3: {result['recall@3']:.3f} | Recall@5: {result['recall@5']:.3f} | Recall@10: {result['recall@10']:.3f}")
            print(f"   Question: {result['question'][:150]}...")
        
        print("\n" + "="*100)
    
    # Print top 10 for each recall metric
    print_top_10("recall@1", query_recalls)
    print_top_10("recall@3", query_recalls)
    print_top_10("recall@5", query_recalls)
    print_top_10("recall@10", query_recalls)
    
    # Statistics summary
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    
    queries_with_zero_recall1 = sum(1 for q in query_recalls if q["recall@1"] == 0)
    queries_with_zero_recall3 = sum(1 for q in query_recalls if q["recall@3"] == 0)
    queries_with_zero_recall5 = sum(1 for q in query_recalls if q["recall@5"] == 0)
    queries_with_zero_recall10 = sum(1 for q in query_recalls if q["recall@10"] == 0)
    
    print(f"\nQueries with Recall@1 = 0: {queries_with_zero_recall1} ({queries_with_zero_recall1/len(query_recalls)*100:.2f}%)")
    print(f"Queries with Recall@3 = 0: {queries_with_zero_recall3} ({queries_with_zero_recall3/len(query_recalls)*100:.2f}%)")
    print(f"Queries with Recall@5 = 0: {queries_with_zero_recall5} ({queries_with_zero_recall5/len(query_recalls)*100:.2f}%)")
    print(f"Queries with Recall@10 = 0: {queries_with_zero_recall10} ({queries_with_zero_recall10/len(query_recalls)*100:.2f}%)")
    
    avg_recall1 = sum(q["recall@1"] for q in query_recalls) / len(query_recalls)
    avg_recall3 = sum(q["recall@3"] for q in query_recalls) / len(query_recalls)
    avg_recall5 = sum(q["recall@5"] for q in query_recalls) / len(query_recalls)
    avg_recall10 = sum(q["recall@10"] for q in query_recalls) / len(query_recalls)
    
    print(f"\nAverage Recall@1: {avg_recall1:.3f}")
    print(f"Average Recall@3: {avg_recall3:.3f}")
    print(f"Average Recall@5: {avg_recall5:.3f}")
    print(f"Average Recall@10: {avg_recall10:.3f}")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    find_low_score_queries()