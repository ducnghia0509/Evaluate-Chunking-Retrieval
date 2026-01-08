import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import tiktoken
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from log.chunking_logger import ChunkingLogger


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_by_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text by token count with overlap"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        start = end - overlap
        if start >= len(tokens):
            break
    
    return chunks


def parent_child_chunking(markdown_file: Path, config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Parent-child chunking strategy
    - Parent chunks: larger context chunks
    - Child chunks: smaller retrieval chunks linked to parents
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parent_chunk_size = config.get('parent_chunk_size', 1000)
    child_chunk_size = config.get('child_chunk_size', 250)
    child_overlap = config.get('child_overlap', 50)
    link_strategy = config.get('link_strategy', 'id_mapping')
    
    # Create parent chunks
    parent_chunks_text = chunk_by_tokens(content, parent_chunk_size, child_overlap)
    
    parent_chunks = []
    child_chunks = []
    
    for parent_idx, parent_text in enumerate(parent_chunks_text):
        parent_id = f"{markdown_file.stem}_parent_{parent_idx}"
        
        # Create parent chunk
        parent_chunks.append({
            'chunk_id': parent_id,
            'source_file': str(markdown_file.name),
            'chunk_index': parent_idx,
            'chunk_type': 'parent',
            'text': parent_text.strip(),
            'metadata': {
                'strategy': 'parent_child',
                'role': 'parent',
                'token_count': count_tokens(parent_text),
                'child_count': 0  # Will be updated
            }
        })
        
        # Create child chunks from this parent
        child_chunks_text = chunk_by_tokens(parent_text, child_chunk_size, child_overlap)
        
        for child_idx, child_text in enumerate(child_chunks_text):
            child_id = f"{markdown_file.stem}_child_{parent_idx}_{child_idx}"
            
            child_chunks.append({
                'chunk_id': child_id,
                'source_file': str(markdown_file.name),
                'chunk_index': len(child_chunks),
                'chunk_type': 'child',
                'text': child_text.strip(),
                'metadata': {
                    'strategy': 'parent_child',
                    'role': 'child',
                    'parent_id': parent_id,
                    'parent_index': parent_idx,
                    'child_index_in_parent': child_idx,
                    'token_count': count_tokens(child_text)
                }
            })
        
        # Update parent's child count
        parent_chunks[-1]['metadata']['child_count'] = len(child_chunks_text)
    
    return parent_chunks, child_chunks


def process_all_files(config_path: str = "../../Chunking/config.json"):
    """Process all markdown files with parent-child chunking"""
    # Initialize logger
    logger = ChunkingLogger('parent_child')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        chunking_config = config['chunking_strategies']['parent_child']
        markdown_dir = Path(config['paths']['markdown_dir'])
        output_dir = Path(config['paths']['output_dir']) / 'parent_child'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.logger.info(f"Config loaded from: {config_path}")
        logger.logger.info(f"Markdown directory: {markdown_dir}")
        logger.logger.info(f"Output directory: {output_dir}")
        logger.logger.info(f"Config: {json.dumps(chunking_config, indent=2)}")
        
        all_parent_chunks = []
        all_child_chunks = []
        
        # Get all markdown files
        md_files = list(markdown_dir.rglob('*.md'))
        logger.logger.info(f"Found {len(md_files)} markdown files to process")
        
        # Process all markdown files
        for idx, md_file in enumerate(md_files, 1):
            try:
                logger.log_file_start(str(md_file))
                logger.logger.info(f"[{idx}/{len(md_files)}] Processing: {md_file.name}")
                
                parent_chunks, child_chunks = parent_child_chunking(md_file, chunking_config)
                all_parent_chunks.extend(parent_chunks)
                all_child_chunks.extend(child_chunks)
                
                # Log chunk statistics for both parent and child chunks
                logger.log_chunk_stats(parent_chunks)
                logger.log_chunk_stats(child_chunks)
                
                # Calculate file-specific stats
                parent_tokens = [c['metadata']['token_count'] for c in parent_chunks]
                child_tokens = [c['metadata']['token_count'] for c in child_chunks]
                child_counts = [c['metadata']['child_count'] for c in parent_chunks]
                
                file_stats = {
                    'num_parent_chunks': len(parent_chunks),
                    'num_child_chunks': len(child_chunks),
                    'parent_stats': {
                        'total_tokens': sum(parent_tokens),
                        'avg_tokens': sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0,
                        'min_tokens': min(parent_tokens) if parent_tokens else 0,
                        'max_tokens': max(parent_tokens) if parent_tokens else 0,
                        'avg_children_per_parent': sum(child_counts) / len(child_counts) if child_counts else 0
                    },
                    'child_stats': {
                        'total_tokens': sum(child_tokens),
                        'avg_tokens': sum(child_tokens) / len(child_tokens) if child_tokens else 0,
                        'min_tokens': min(child_tokens) if child_tokens else 0,
                        'max_tokens': max(child_tokens) if child_tokens else 0
                    }
                }
                
                logger.log_file_complete(
                    str(md_file), 
                    len(parent_chunks) + len(child_chunks), 
                    file_stats
                )
                
                # Save chunks for this file
                relative_path = md_file.relative_to(markdown_dir)
                output_file_parent = output_dir / relative_path.parent / f"{md_file.stem}_parent_chunks.json"
                output_file_child = output_dir / relative_path.parent / f"{md_file.stem}_child_chunks.json"
                output_file_parent.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file_parent, 'w', encoding='utf-8') as f:
                    json.dump(parent_chunks, f, ensure_ascii=False, indent=2)
                
                with open(output_file_child, 'w', encoding='utf-8') as f:
                    json.dump(child_chunks, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.log_error(str(md_file), e)
                continue
        
        # Save all chunks combined
        all_parent_file = output_dir / "all_parent_chunks.json"
        all_child_file = output_dir / "all_child_chunks.json"
        
        with open(all_parent_file, 'w', encoding='utf-8') as f:
            json.dump(all_parent_chunks, f, ensure_ascii=False, indent=2)
        
        with open(all_child_file, 'w', encoding='utf-8') as f:
            json.dump(all_child_chunks, f, ensure_ascii=False, indent=2)
        
        logger.logger.info(f"Parent chunks saved to: {all_parent_file}")
        logger.logger.info(f"Child chunks saved to: {all_child_file}")
        
    finally:
        # Finalize logging and save statistics
        logger.finalize()


if __name__ == "__main__":
    process_all_files()
