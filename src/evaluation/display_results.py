"""
Display Historical Evaluation Results
Hiển thị các plot/chart từ các file kết quả trong folder Evaluate/results
Logic tương tự file run_evaluation.py
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import pandas as pd
import plotly.graph_objects as go
import re

# Configuration
RESULTS_DIR = "../../Evaluate/results"


def parse_filename(filename: str) -> Dict:
    """
    Parse evaluation filename to extract metadata
    Format: eval_{chunking}_{retrieval}_{timestamp}.json
    
    Args:
        filename: Name of the evaluation file
        
    Returns:
        Dict with chunking_strategy, retrieval_method, timestamp
    """
    # Remove .json extension
    name = filename.replace('.json', '')
    
    # Pattern: eval_{chunking}_{retrieval}_{timestamp}
    # timestamp format: YYYYMMDD_HHMMSS
    pattern = r'eval_(.+?)_(.+?)_(\d{8}_\d{6})$'
    match = re.match(pattern, name)
    
    if match:
        chunking = match.group(1)
        retrieval = match.group(2)
        timestamp_str = match.group(3)
        
        # Parse timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except:
            timestamp = None
        
        return {
            'chunking_strategy': chunking,
            'retrieval_method': retrieval,
            'timestamp': timestamp,
            'timestamp_str': timestamp_str
        }
    
    return None


def load_historical_results(results_dir: str = RESULTS_DIR) -> Dict:
    """
    Load all historical results from folder
    If duplicate (same chunking + retrieval), keep the newest one
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dict of historical results
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"⚠️ Results directory not found: {results_path}")
        return {}
    
    # Get all JSON files
    json_files = list(results_path.glob("eval_*.json"))
    print(f"📁 Found {len(json_files)} evaluation files")
    
    # Parse and load each file
    file_metadata = []
    for json_file in json_files:
        metadata = parse_filename(json_file.name)
        if metadata:
            metadata['filepath'] = json_file
            file_metadata.append(metadata)
    
    # Sort by timestamp (newest first)
    file_metadata.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)
    
    # Keep only newest for each chunking + retrieval combination
    seen_combinations = set()
    filtered_files = []
    
    for meta in file_metadata:
        combination = f"{meta['chunking_strategy']}+{meta['retrieval_method']}"
        if combination not in seen_combinations:
            seen_combinations.add(combination)
            filtered_files.append(meta)
            print(f"✓ {combination}: {meta['timestamp_str']}")
        else:
            print(f"  ⏭️ Skipped older: {combination} ({meta['timestamp_str']})")
    
    # Load data from filtered files
    historical_results = {}
    for meta in filtered_files:
        try:
            with open(meta['filepath'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            combination = f"{meta['chunking_strategy']}+{meta['retrieval_method']}"
            historical_results[combination] = {
                'chunking_strategy': meta['chunking_strategy'],
                'retrieval_method': meta['retrieval_method'],
                'timestamp': meta['timestamp'],
                'timestamp_str': meta['timestamp_str'],
                'summary': data.get('summary', {}),
                'results': data.get('results', [])
            }
            
            print(f"✅ Loaded {combination}: {len(data.get('results', []))} queries")
        except Exception as e:
            print(f"❌ Error loading {meta['filepath']}: {e}")
    
    print(f"\n📊 Total loaded: {len(historical_results)} unique combinations")
    return historical_results


def generate_comparison_heatmap(historical_results: Dict) -> go.Figure:
    """
    Generate comparison heatmap - màu được áp dụng theo từng hàng (row-wise)
    Mỗi hàng = 1 metric, so sánh giữa các strategies trên cùng hàng
    
    Args:
        historical_results: Dict of loaded results
        
    Returns:
        Plotly figure
    """
    if not historical_results:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No historical results found",
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
    
    # Prepare data
    strategies = list(historical_results.keys())
    
    # Get metrics from first result
    first_key = strategies[0]
    all_metrics = list(historical_results[first_key]['summary'].keys())
    
    # Create data matrix: rows = metrics, cols = strategies
    data_matrix = []
    for metric in all_metrics:
        row = []
        for strategy in strategies:
            val = historical_results[strategy]['summary'].get(metric, 0)
            row.append(val)
        data_matrix.append(row)
    
    # CHỈNH SỬA QUAN TRỌNG: Tính màu theo từng hàng
    # Cách 1: Chuẩn hóa từng hàng (row-wise normalization)
    normalized_matrix = []
    for row in data_matrix:
        row_min = min(row)
        row_max = max(row)
        if row_max - row_min > 0:
            normalized_row = [(val - row_min) / (row_max - row_min) for val in row]
        else:
            normalized_row = [0.5 for _ in row]  # Tất cả bằng nhau
        normalized_matrix.append(normalized_row)
    
    # Tạo heatmap với normalized matrix
    fig = go.Figure(data=go.Heatmap(
        z=normalized_matrix,  # Dùng ma trận đã chuẩn hóa theo hàng
        x=strategies,
        y=all_metrics,
        colorscale='RdYlGn',
        text=[[f"{val:.4f}" for val in row] for row in data_matrix],  # Hiển thị giá trị thực
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Normalized Score"),
        hoverongaps=False,
        showscale=True,
        # Quan trọng: chỉ định zmin và zmax cố định
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title="📊 Comparison Heatmap - All Strategies (Row-wise Comparison)",
        xaxis_title="Strategy (Chunking + Retrieval)",
        yaxis_title="Metrics",
        height=max(400, len(all_metrics) * 60),
        width=max(800, len(strategies) * 120),
        xaxis=dict(tickangle=-45),
        font=dict(size=11),
        margin=dict(l=150, r=100, t=80, b=150)
    )
    
    # Thêm annotation để giải thích
    fig.add_annotation(
        text="💡 Màu sắc được so sánh theo từng hàng (cùng metric)",
        xref="paper", yref="paper",
        x=0.5, y=-0.15, showarrow=False,
        font=dict(size=10, color="gray"),
        align="center"
    )
    
    return fig

def generate_strategy_summary_table(historical_results: Dict) -> pd.DataFrame:
    """
    Generate summary table (giống logic trong run_evaluation.py)
    
    Args:
        historical_results: Dict of loaded results
        
    Returns:
        DataFrame with summary
    """
    if not historical_results:
        return pd.DataFrame()
    
    # Create summary data
    summary_data = []
    for strategy_name, strategy_data in historical_results.items():
        metrics = strategy_data.get('summary', {})
        row = {
            'Strategy': strategy_name,
            'Timestamp': strategy_data['timestamp_str'],
            'Recall@10': f"{metrics.get('recall@10', 0):.4f}",
            'Precision@10': f"{metrics.get('precision@10', 0):.4f}",
            'MRR': f"{metrics.get('mrr', 0):.4f}",
            'NDCG@10': f"{metrics.get('ndcg@10', 0):.4f}",
            'Queries': len(strategy_data['results'])
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    print("="*80)
    print("📊 DISPLAYING HISTORICAL EVALUATION RESULTS")
    print("="*80)
    print()
    
    # Load all results
    results = load_historical_results()
    
    if not results:
        print("\n⚠️ No results found!")
    else:
        print("\n" + "="*80)
        print("📈 GENERATING PLOTS")
        print("="*80)
        print()
        
        # Generate summary table
        summary_table = generate_strategy_summary_table(results)
        print("📋 Summary Table:")
        print(summary_table.to_string(index=False))
        print()
        
        # Save summary table to CSV
        output_dir = Path(RESULTS_DIR)
        table_file = output_dir / "comparison_summary.csv"
        summary_table.to_csv(table_file, index=False, encoding='utf-8-sig')
        print(f"💾 Summary table saved to: {table_file.absolute()}")
        print()
        
        # Generate and save heatmap
        print("🔥 Generating comparison heatmap...")
        heatmap = generate_comparison_heatmap(results)
        
        # Save as PNG only
        png_file = output_dir / "comparison_heatmap.png"
        try:
            heatmap.write_image(str(png_file))
            print(f"💾 Heatmap saved to: {png_file.absolute()}")
        except Exception as e:
            print(f"❌ Error saving PNG: {e}")
            print("💡 Please install kaleido: pip install kaleido")
        
        print("\n" + "="*80)
        print("✨ DONE!")
        print("="*80)
        print(f"\n📂 Output files saved in: {output_dir.absolute()}")
        print(f"   - comparison_summary.csv")
        print(f"   - comparison_heatmap.png")
