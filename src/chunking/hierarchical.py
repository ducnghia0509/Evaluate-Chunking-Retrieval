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


def extract_sections(content: str) -> List[Tuple[str, str, int]]:
    """
    Extract sections based on markdown headers
    Returns: [(header_text, content, level)]
    """
    sections = []
    lines = content.split('\n')
    
    current_header = None
    current_level = 0
    current_content = []
    
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            # Save previous section
            if current_header is not None:
                sections.append((
                    current_header,
                    '\n'.join(current_content).strip(),
                    current_level
                ))
            
            # Start new section
            current_level = len(header_match.group(1))
            current_header = header_match.group(2)
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_header is not None:
        sections.append((
            current_header,
            '\n'.join(current_content).strip(),
            current_level
        ))
    
    return sections


def split_by_paragraphs(text: str, max_size: int) -> List[str]:
    """Split text into paragraphs"""
    paragraphs = re.split(r'\n\n+', text)
    result = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if count_tokens(para) <= max_size:
            result.append(para)
        else:
            # Further split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            
            for sent in sentences:
                test = current + " " + sent if current else sent
                if count_tokens(test) <= max_size:
                    current = test
                else:
                    if current:
                        result.append(current)
                    current = sent
            
            if current:
                result.append(current)
    
    return result


def hierarchical_chunking(markdown_file: Path, config: Dict) -> List[Dict]:
    """
    Hierarchical chunking strategy
    Creates chunks at multiple levels: section → paragraph → sentence
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    max_chunk_size = config.get('max_chunk_size', 800)
    min_chunk_size = config.get('min_chunk_size', 150)
    store_hierarchy = config.get('store_hierarchy', True)
    
    chunks = []
    chunk_idx = 0
    
    # Extract sections
    sections = extract_sections(content)
    
    if not sections:
        # No headers, treat entire document as one section
        sections = [("Document", content, 0)]
    
    for section_idx, (header, section_content, level) in enumerate(sections):
        if not section_content.strip():
            continue
        
        # Check if entire section fits in one chunk
        section_tokens = count_tokens(section_content)
        
        if section_tokens <= max_chunk_size:
            # Entire section is one chunk
            chunks.append({
                'chunk_id': f"{markdown_file.stem}_chunk_{chunk_idx}",
                'source_file': str(markdown_file.name),
                'chunk_index': chunk_idx,
                'text': section_content.strip(),
                'metadata': {
                    'strategy': 'hierarchical',
                    'level': 'section',
                    'section_header': header,
                    'section_level': level,
                    'token_count': section_tokens,
                    'hierarchy': {
                        'section_index': section_idx,
                        'parent_section': header
                    } if store_hierarchy else None
                }
            })
            chunk_idx += 1
        else:
            # Split section into paragraphs
            paragraphs = split_by_paragraphs(section_content, max_chunk_size)
            
            for para_idx, para in enumerate(paragraphs):
                if count_tokens(para) >= min_chunk_size or para_idx == len(paragraphs) - 1:
                    chunks.append({
                        'chunk_id': f"{markdown_file.stem}_chunk_{chunk_idx}",
                        'source_file': str(markdown_file.name),
                        'chunk_index': chunk_idx,
                        'text': para.strip(),
                        'metadata': {
                            'strategy': 'hierarchical',
                            'level': 'paragraph',
                            'section_header': header,
                            'section_level': level,
                            'token_count': count_tokens(para),
                            'hierarchy': {
                                'section_index': section_idx,
                                'paragraph_index': para_idx,
                                'parent_section': header
                            } if store_hierarchy else None
                        }
                    })
                    chunk_idx += 1
    
    return chunks


def process_all_files(config_path: str = "../../Chunking/config.json"):
    """Process all markdown files with hierarchical chunking"""
    # Initialize logger
    logger = ChunkingLogger('hierarchical')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        chunking_config = config['chunking_strategies']['hierarchical']
        markdown_dir = Path(config['paths']['markdown_dir'])
        output_dir = Path(config['paths']['output_dir']) / 'hierarchical'
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
                
                chunks = hierarchical_chunking(md_file, chunking_config)
                all_chunks.extend(chunks)
                
                # Log chunk statistics for this file
                logger.log_chunk_stats(chunks)
                
                # Calculate file-specific stats
                token_counts = [c['metadata']['token_count'] for c in chunks]
                hierarchy_levels = [c['metadata']['hierarchy_level'] for c in chunks]
                file_stats = {
                    'num_chunks': len(chunks),
                    'total_tokens': sum(token_counts),
                    'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
                    'min_tokens': min(token_counts) if token_counts else 0,
                    'max_tokens': max(token_counts) if token_counts else 0,
                    'hierarchy_levels': {
                        'min': min(hierarchy_levels) if hierarchy_levels else 0,
                        'max': max(hierarchy_levels) if hierarchy_levels else 0,
                        'avg': sum(hierarchy_levels) / len(hierarchy_levels) if hierarchy_levels else 0
                    }
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
