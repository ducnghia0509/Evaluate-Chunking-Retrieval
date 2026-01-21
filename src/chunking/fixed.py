import json
import os
from pathlib import Path
from typing import List, Dict
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
        
        # Move start position with overlap
        start = end - overlap
        
        if start >= len(tokens):
            break
    
    return chunks


def fixed_chunking(markdown_file: Path, config: Dict) -> List[Dict]:
    """
    Fixed size chunking strategy
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunk_size = config.get('chunk_size', 512)
    chunk_overlap = config.get('chunk_overlap', 128)
    unit = config.get('unit', 'token')
    
    if unit == 'token':
        chunk_texts = chunk_by_tokens(content, chunk_size, chunk_overlap)
    else:
        # Character-based chunking
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
    
    # Create chunk metadata
    chunks = []
    for idx, chunk_text in enumerate(chunk_texts):
        chunks.append({
            'chunk_id': f"{markdown_file.stem}_chunk_{idx}",
            'source_file': str(markdown_file.name),
            'chunk_index': idx,
            'text': chunk_text.strip(),
            'metadata': {
                'strategy': 'fixed',
                'chunk_size': chunk_size,
                'overlap': chunk_overlap,
                'token_count': count_tokens(chunk_text)
            }
        })
    
    # Merge chunks with token count < 50 into previous chunk
    min_token_threshold = 50
    merged_chunks = []
    
    for i, chunk in enumerate(chunks):
        token_count = chunk['metadata']['token_count']
        
        # If chunk is too small and there's a previous chunk, merge it
        if token_count < min_token_threshold and merged_chunks:
            # Merge with previous chunk
            prev_chunk = merged_chunks[-1]
            prev_chunk['text'] = prev_chunk['text'] + ' ' + chunk['text']
            prev_chunk['metadata']['token_count'] = count_tokens(prev_chunk['text'])
            prev_chunk['metadata']['merged'] = prev_chunk['metadata'].get('merged', 0) + 1
        else:
            merged_chunks.append(chunk)
    
    # Re-index chunks after merging
    for idx, chunk in enumerate(merged_chunks):
        chunk['chunk_id'] = f"{markdown_file.stem}_chunk_{idx}"
        chunk['chunk_index'] = idx
    
    return merged_chunks


def process_all_files(config_path: str = "../../Chunking/config.json"):
    """Process all markdown files with fixed chunking"""
    # Initialize logger
    logger = ChunkingLogger('fixed')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        chunking_config = config['chunking_strategies']['fixed']
        markdown_dir = Path(config['paths']['markdown_dir'])
        output_dir = Path(config['paths']['output_dir']) / 'fixed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.logger.info(f"Config loaded from: {config_path}")
        logger.logger.info(f"Markdown directory: {markdown_dir}")
        logger.logger.info(f"Output directory: {output_dir}")
        logger.logger.info(f"Config: {json.dumps(chunking_config, indent=2)}")
        
        all_chunks = []
        
        # Get all markdown files
        md_files = list(markdown_dir.rglob('*.md'))
        logger.logger.info(f"Found {len(md_files)} markdown files to process")
        
        # Process all markdown files
        for idx, md_file in enumerate(md_files, 1):
            try:
                logger.log_file_start(str(md_file))
                logger.logger.info(f"[{idx}/{len(md_files)}] Processing: {md_file.name}")
                
                chunks = fixed_chunking(md_file, chunking_config)
                all_chunks.extend(chunks)
                
                # Log chunk statistics for this file
                logger.log_chunk_stats(chunks)
                
                # Calculate file-specific stats
                token_counts = [c['metadata']['token_count'] for c in chunks]
                file_stats = {
                    'num_chunks': len(chunks),
                    'total_tokens': sum(token_counts),
                    'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
                    'min_tokens': min(token_counts) if token_counts else 0,
                    'max_tokens': max(token_counts) if token_counts else 0,
                    'chunk_size_config': chunking_config.get('chunk_size', 512),
                    'overlap_config': chunking_config.get('chunk_overlap', 128)
                }
                
                logger.log_file_complete(str(md_file), len(chunks), file_stats)
                
                # Save chunks for this file
                relative_path = md_file.relative_to(markdown_dir)
                output_file = output_dir / relative_path.parent / f"{md_file.stem}_chunks.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.log_error(str(md_file), e)
                continue
        
        # Save all chunks combined
        all_chunks_file = output_dir / "all_chunks.json"
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.logger.info(f"All chunks saved to: {all_chunks_file}")
        
    finally:
        # Finalize logging and save statistics
        logger.finalize()


if __name__ == "__main__":
    process_all_files()
