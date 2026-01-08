import json
import os
import re
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


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on markdown structure"""
    # Split by double newlines (paragraph separator)
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def merge_small_paragraphs(paragraphs: List[str], min_size: int) -> List[str]:
    """Merge paragraphs that are too small"""
    merged = []
    buffer = ""
    
    for para in paragraphs:
        token_count = count_tokens(buffer + " " + para if buffer else para)
        
        if buffer and token_count > min_size:
            # Buffer is large enough, save it
            merged.append(buffer)
            buffer = para
        else:
            # Add to buffer
            buffer = buffer + "\n\n" + para if buffer else para
    
    if buffer:
        merged.append(buffer)
    
    return merged


def split_large_paragraphs(paragraphs: List[str], max_size: int) -> List[str]:
    """Split paragraphs that are too large"""
    result = []
    
    for para in paragraphs:
        if count_tokens(para) <= max_size:
            result.append(para)
        else:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            
            for sentence in sentences:
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if count_tokens(test_chunk) <= max_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:
                result.append(current_chunk)
    
    return result


def structure_paragraph_chunking(markdown_file: Path, config: Dict) -> List[Dict]:
    """
    Structure-aware paragraph chunking strategy
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    max_chunk_size = config.get('max_chunk_size', 600)
    min_chunk_size = config.get('min_chunk_size', 100)
    merge_short = config.get('merge_short_paragraphs', True)
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(content)
    
    # Merge short paragraphs if enabled
    if merge_short:
        paragraphs = merge_small_paragraphs(paragraphs, min_chunk_size)
    
    # Split large paragraphs
    paragraphs = split_large_paragraphs(paragraphs, max_chunk_size)
    
    # Create chunk metadata
    chunks = []
    for idx, para in enumerate(paragraphs):
        # Extract header if paragraph starts with one
        header_match = re.match(r'^(#{1,6})\s+(.+)$', para.split('\n')[0])
        header = header_match.group(2) if header_match else None
        
        chunks.append({
            'chunk_id': f"{markdown_file.stem}_chunk_{idx}",
            'source_file': str(markdown_file.name),
            'chunk_index': idx,
            'text': para.strip(),
            'metadata': {
                'strategy': 'structure_paragraph',
                'header': header,
                'token_count': count_tokens(para),
                'has_header': header is not None
            }
        })
    
    return chunks


def process_all_files(config_path: str = "../../Chunking/config.json"):
    """Process all markdown files with structure paragraph chunking"""
    # Initialize logger
    logger = ChunkingLogger('structure_paragraph')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        chunking_config = config['chunking_strategies']['structure_paragraph']
        markdown_dir = Path(config['paths']['markdown_dir'])
        output_dir = Path(config['paths']['output_dir']) / 'structure_paragraph'
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
                
                chunks = structure_paragraph_chunking(md_file, chunking_config)
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
                    'chunks_with_header': sum(1 for c in chunks if c['metadata'].get('has_header', False))
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
