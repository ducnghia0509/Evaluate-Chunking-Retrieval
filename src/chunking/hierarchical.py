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


def extract_complete_sections_with_header(content: str) -> List[Tuple[str, str, int, str, str]]:
    """
    Extract complete sections WITH header line included in content
    Returns: [(header_text, full_header_line, complete_content, level, full_path)]
    """
    sections = []
    lines = content.split('\n')
    
    current_header_text = None
    current_full_header = None
    current_level = 0
    current_content_lines = []  # Bao gồm cả header line
    header_stack = []
    
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            # Save previous section
            if current_header_text is not None:
                full_path = " > ".join(header_stack + [current_header_text]) if header_stack else current_header_text
                # Content BAO GỒM header line
                complete_content = '\n'.join(current_content_lines).strip()
                sections.append((
                    current_header_text,      # Chỉ text
                    current_full_header,      # Full line với #
                    complete_content,         # Toàn bộ content có header
                    current_level,
                    full_path
                ))
            
            # Update header stack
            new_level = len(header_match.group(1))
            new_header_text = header_match.group(2)
            new_full_header = line  # Giữ nguyên cả dấu #
            
            # Pop headers với level cao hơn hoặc bằng
            while header_stack and len(header_stack) >= new_level:
                header_stack.pop()
            
            # Start new section
            current_level = new_level
            current_header_text = new_header_text
            current_full_header = new_full_header
            header_stack.append(new_header_text)
            current_content_lines = [new_full_header]  # BẮT ĐẦU với header line!
        else:
            current_content_lines.append(line)
    
    # Save last section
    if current_header_text is not None:
        full_path = " > ".join(header_stack) if header_stack else current_header_text
        complete_content = '\n'.join(current_content_lines).strip()
        sections.append((
            current_header_text,
            current_full_header,
            complete_content,
            current_level,
            full_path
        ))
    elif current_content_lines:
        # No headers in document
        complete_content = '\n'.join(current_content_lines).strip()
        sections.append((
            "Document",
            "# Document",
            complete_content,
            0,
            "Document"
        ))
    
    return sections


def ensure_header_in_chunk(chunk_text: str, full_header_line: str, 
                          header_text: str, level: int,
                          is_first_chunk: bool = True) -> str:
    """
    Ensure header is included in chunk text when appropriate
    """
    # Nếu chunk đã bắt đầu bằng header, giữ nguyên
    if chunk_text.strip().startswith('#'):
        return chunk_text
    
    # Nếu là chunk đầu tiên của section, thêm header
    if is_first_chunk and full_header_line:
        # Kiểm tra xem thêm header có làm chunk quá lớn không
        combined = f"{full_header_line}\n\n{chunk_text}"
        # Ước tính token - nếu quá lớn, có thể chỉ thêm header text
        if count_tokens(combined) <= 800:  # Giả định max size
            return combined
        else:
            # Chỉ thêm header text không dấu #
            return f"{header_text}\n\n{chunk_text}"
    
    return chunk_text


def smart_split_section_with_header(complete_content: str, full_header_line: str,
                                  header_text: str, level: int,
                                  max_size: int, min_size: int) -> List[Tuple[str, bool]]:
    """
    Split section while preserving header in first chunk
    Returns: [(chunk_text, has_header)]
    """
    # Nếu toàn bộ section đủ nhỏ, giữ nguyên
    if count_tokens(complete_content) <= max_size:
        return [(complete_content, True)]
    
    # Tách thành paragraphs giữ nguyên cấu trúc
    lines = complete_content.split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        line_stripped = line.strip()
        # Dòng trống đánh dấu kết thúc paragraph
        if not line_stripped:
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
        else:
            current_para.append(line)
    
    if current_para:
        paragraphs.append('\n'.join(current_para))
    
    # Xử lý paragraphs
    chunks = []
    current_chunk_lines = []
    current_tokens = 0
    
    for i, para in enumerate(paragraphs):
        para_tokens = count_tokens(para)
        
        # Paragraph riêng lẻ quá lớn
        if para_tokens > max_size:
            # Lưu chunk hiện tại
            if current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = []
                current_tokens = 0
            
            # Chia paragraph lớn
            if para.startswith('#'):
                # Header paragraph, không chia
                chunks.append(para)
            else:
                # Content paragraph, chia theo sentences
                sub_chunks = split_large_content(para, max_size)
                chunks.extend(sub_chunks)
        
        # Có thể thêm vào chunk hiện tại
        elif current_tokens + para_tokens <= max_size:
            current_chunk_lines.append(para)
            current_tokens += para_tokens
        
        # Chunk hiện tại đầy
        else:
            if current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
            
            current_chunk_lines = [para]
            current_tokens = para_tokens
    
    # Thêm chunk cuối
    if current_chunk_lines:
        chunks.append('\n'.join(current_chunk_lines))
    
    # Gộp các chunks nhỏ
    merged_chunks = merge_small_chunks_with_header(chunks, max_size, min_size)
    
    # Đảm bảo chunk đầu tiên có header
    result = []
    for i, chunk in enumerate(merged_chunks):
        has_header = chunk.strip().startswith('#')
        
        # Nếu là chunk đầu tiên và chưa có header
        if i == 0 and not has_header and full_header_line:
            chunk_with_header = f"{full_header_line}\n\n{chunk}"
            # Kiểm tra size
            if count_tokens(chunk_with_header) <= max_size:
                result.append((chunk_with_header, True))
            else:
                result.append((chunk, False))
        else:
            result.append((chunk, has_header))
    
    return result


def split_large_content(text: str, max_size: int) -> List[str]:
    """Split large content by sentences"""
    # Ưu tiên split bằng newlines
    if '\n' in text:
        lines = text.split('\n')
        chunks = []
        current = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens <= max_size:
                current.append(line)
                current_tokens += line_tokens
            else:
                if current:
                    chunks.append('\n'.join(current))
                current = [line]
                current_tokens = line_tokens
        
        if current:
            chunks.append('\n'.join(current))
        
        return chunks
    
    # Split bằng sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    chunks = []
    current = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens <= max_size:
            current.append(sent)
            current_tokens += sent_tokens
        else:
            if current:
                chunks.append(' '.join(current))
            current = [sent]
            current_tokens = sent_tokens
    
    if current:
        chunks.append(' '.join(current))
    
    return chunks


def merge_small_chunks_with_header(chunks: List[str], max_size: int, min_size: int) -> List[str]:
    """Merge small chunks with header awareness"""
    if not chunks:
        return []
    
    result = []
    current = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        
        # Bỏ qua chunks quá nhỏ (sẽ gộp sau)
        if chunk_tokens < 10 and not chunk.startswith('#'):
            continue
        
        # Nếu chunk có header, xử lý đặc biệt
        if chunk.startswith('#'):
            # Lưu chunk hiện tại nếu có
            if current:
                result.append('\n\n'.join(current))
                current = []
                current_tokens = 0
            
            # Header chunk đứng riêng
            result.append(chunk)
        elif current_tokens + chunk_tokens <= max_size:
            current.append(chunk)
            current_tokens += chunk_tokens
        else:
            if current:
                # Kiểm tra nếu chunk hiện tại quá nhỏ
                if current_tokens < min_size and result:
                    # Gộp với chunk trước
                    last_chunk = result[-1]
                    last_tokens = count_tokens(last_chunk)
                    
                    if last_tokens + current_tokens <= max_size:
                        result[-1] = last_chunk + '\n\n' + '\n\n'.join(current)
                    else:
                        result.append('\n\n'.join(current))
                else:
                    result.append('\n\n'.join(current))
            
            current = [chunk]
            current_tokens = chunk_tokens
    
    # Xử lý chunk cuối
    if current:
        if current_tokens < min_size and result:
            last_chunk = result[-1]
            last_tokens = count_tokens(last_chunk)
            
            if last_tokens + current_tokens <= max_size:
                result[-1] = last_chunk + '\n\n' + '\n\n'.join(current)
            else:
                result.append('\n\n'.join(current))
        else:
            result.append('\n\n'.join(current))
    
    return result


def hierarchical_chunking(markdown_file: Path, config: Dict) -> List[Dict]:
    """
    FIXED hierarchical chunking with header preservation
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get configuration
    levels = config.get('levels', ['section', 'paragraph', 'sentence'])
    max_chunk_size = config.get('max_chunk_size', 800)
    min_chunk_size = config.get('min_chunk_size', 150)
    store_hierarchy = config.get('store_hierarchy', True)
    
    effective_min_size = max(min_chunk_size, 50)
    
    chunks = []
    chunk_idx = 0
    
    # Extract sections WITH header in content
    sections = extract_complete_sections_with_header(content)
    
    if not sections:
        return chunks
    
    for section_idx, (header_text, full_header_line, complete_content, level, full_path) in enumerate(sections):
        if not complete_content.strip():
            continue
        
        section_tokens = count_tokens(complete_content)
        
        # Skip very small sections
        if section_tokens < 20:
            continue
        
        # LEVEL 1: Section chunk (giữ nguyên cả header)
        if 'section' in levels and should_create_section_chunk(complete_content, max_chunk_size, effective_min_size):
            chunks.append({
                'chunk_id': f"{markdown_file.stem}_chunk_{chunk_idx}",
                'source_file': str(markdown_file.name),
                'chunk_index': chunk_idx,
                'text': complete_content.strip(),
                'metadata': {
                    'strategy': 'hierarchical',
                    'level': 'section',
                    'section_header': header_text,
                    'full_header_line': full_header_line,
                    'full_section_path': full_path,
                    'header_level': level,
                    'token_count': section_tokens,
                    'has_header_in_text': True,
                    'hierarchy': {
                        'section_index': section_idx,
                        'parent_path': full_path,
                        'chunking_level': 'section',
                        'was_merged': False
                    } if store_hierarchy else None,
                    'chunking_info': {
                        'max_chunk_size': max_chunk_size,
                        'min_chunk_size': effective_min_size,
                        'levels': levels
                    }
                }
            })
            chunk_idx += 1
            continue
        
        # LEVEL 2/3: Split section với header preservation
        section_chunks = smart_split_section_with_header(
            complete_content, full_header_line, header_text, level,
            max_chunk_size, effective_min_size
        )
        
        for chunk_in_section_idx, (chunk_text, has_header) in enumerate(section_chunks):
            if not chunk_text.strip():
                continue
            
            chunk_tokens = count_tokens(chunk_text)
            
            # Xác định level
            if has_header and '\n\n' not in chunk_text:
                chunk_level = 'section'
            elif chunk_tokens > max_chunk_size * 0.7:  # Chunk lớn
                chunk_level = 'paragraph'
            else:
                # Đếm số paragraphs trong chunk
                paragraphs_in_chunk = len(chunk_text.split('\n\n'))
                if paragraphs_in_chunk > 1:
                    chunk_level = 'section'
                else:
                    chunk_level = 'paragraph'
            
            chunks.append({
                'chunk_id': f"{markdown_file.stem}_chunk_{chunk_idx}",
                'source_file': str(markdown_file.name),
                'chunk_index': chunk_idx,
                'text': chunk_text.strip(),
                'metadata': {
                    'strategy': 'hierarchical',
                    'level': chunk_level,
                    'section_header': header_text,
                    'full_header_line': full_header_line,
                    'full_section_path': full_path,
                    'header_level': level,
                    'token_count': chunk_tokens,
                    'has_header_in_text': has_header,
                    'hierarchy': {
                        'section_index': section_idx,
                        'chunk_in_section_index': chunk_in_section_idx,
                        'total_chunks_in_section': len(section_chunks),
                        'parent_path': full_path,
                        'chunking_level': chunk_level,
                        'was_split': len(section_chunks) > 1
                    } if store_hierarchy else None,
                    'chunking_info': {
                        'max_chunk_size': max_chunk_size,
                        'min_chunk_size': effective_min_size,
                        'levels': levels
                    }
                }
            })
            chunk_idx += 1
    
    # Post-processing merge (giữ nguyên)
    final_chunks = merge_adjacent_small_chunks(chunks, max_chunk_size, effective_min_size)
    
    return final_chunks


def merge_adjacent_small_chunks(chunks: List[Dict], max_size: int, min_size: int) -> List[Dict]:
    """Merge adjacent small chunks while preserving headers"""
    if not chunks:
        return []
    
    final_chunks = []
    i = 0
    
    while i < len(chunks):
        current = chunks[i]
        current_tokens = current['metadata']['token_count']
        
        # Nếu chunk có header, giữ nguyên
        if current['metadata'].get('has_header_in_text', False):
            final_chunks.append(current)
            i += 1
            continue
        
        # Nếu chunk quá nhỏ và không phải cuối
        if current_tokens < min_size and i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            next_tokens = next_chunk['metadata']['token_count']
            
            # Có thể gộp mà không vượt max_size
            if current_tokens + next_tokens <= max_size:
                merged_text = current['text'] + "\n\n" + next_chunk['text']
                merged_tokens = current_tokens + next_tokens
                
                # Giữ header từ chunk có header (nếu có)
                merged_header = current['metadata']['section_header']
                if next_chunk['metadata'].get('has_header_in_text', False):
                    merged_header = next_chunk['metadata']['section_header']
                
                merged_chunk = {
                    'chunk_id': f"{current['source_file'].replace('.md', '')}_merged_{len(final_chunks)}",
                    'source_file': current['source_file'],
                    'chunk_index': len(final_chunks),
                    'text': merged_text.strip(),
                    'metadata': {
                        'strategy': 'hierarchical',
                        'level': 'merged',
                        'section_header': merged_header,
                        'full_section_path': current['metadata'].get('full_section_path', ''),
                        'header_level': current['metadata'].get('header_level', 0),
                        'token_count': merged_tokens,
                        'has_header_in_text': current['metadata'].get('has_header_in_text', False) or 
                                            next_chunk['metadata'].get('has_header_in_text', False),
                        'hierarchy': {
                            'merged_from': [current['chunk_id'], next_chunk['chunk_id']],
                            'original_levels': [current['metadata']['level'], next_chunk['metadata']['level']],
                            'was_merged': True
                        } if current['metadata'].get('hierarchy') else None,
                        'chunking_info': current['metadata']['chunking_info']
                    }
                }
                final_chunks.append(merged_chunk)
                i += 2
            else:
                final_chunks.append(current)
                i += 1
        else:
            final_chunks.append(current)
            i += 1
    
    return final_chunks


# Giữ nguyên process_all_files...

def should_create_section_chunk(section_content: str, max_size: int, min_size: int) -> bool:
    """
    Determine if a section should be kept as one chunk
    """
    tokens = count_tokens(section_content)
    
    # Check if section is within ideal range
    if min_size <= tokens <= max_size:
        return True
    
    # Check if section is slightly larger but has good cohesion
    paragraphs = re.split(r'\n\n+', section_content)
    if len(paragraphs) <= 3 and tokens <= max_size * 1.2:  # Allow 20% overflow for cohesive sections
        # Check if splitting would create very small chunks
        paragraph_tokens = [count_tokens(p) for p in paragraphs]
        if all(t >= min_size * 0.7 for t in paragraph_tokens):  # Paragraphs are reasonably sized
            return False  # Better to split
        else:
            return True  # Keep as section to avoid tiny chunks
    
    return False


def process_all_files(config_path: str = "../../Chunking/config.json"):
    """Process all markdown files with improved hierarchical chunking"""
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
                
                # Log chunk statistics
                if chunks:
                    token_counts = [c['metadata']['token_count'] for c in chunks]
                    levels = [c['metadata']['level'] for c in chunks]
                    
                    level_counts = {}
                    for lvl in levels:
                        level_counts[lvl] = level_counts.get(lvl, 0) + 1
                    
                    min_size = chunking_config.get('min_chunk_size', 150)
                    max_size = chunking_config.get('max_chunk_size', 800)
                    
                    small_chunks = sum(1 for t in token_counts if t < min_size)
                    optimal_chunks = sum(1 for t in token_counts if min_size <= t <= max_size)
                    large_chunks = sum(1 for t in token_counts if t > max_size)
                    
                    file_stats = {
                        'num_chunks': len(chunks),
                        'total_tokens': sum(token_counts),
                        'avg_tokens': sum(token_counts) / len(token_counts),
                        'min_tokens': min(token_counts) if token_counts else 0,
                        'max_tokens': max(token_counts) if token_counts else 0,
                        'levels_distribution': level_counts,
                        'chunk_size_distribution': {
                            'small_chunks': small_chunks,
                            'optimal_chunks': optimal_chunks,
                            'large_chunks': large_chunks
                        }
                    }
                    
                    logger.log_file_complete(str(md_file), len(chunks), file_stats)
                else:
                    logger.logger.warning(f"No chunks created for {md_file.name}")
                
                # Save chunks for this file
                relative_path = md_file.relative_to(markdown_dir)
                output_file = output_dir / relative_path.parent / f"{md_file.stem}_chunks.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.log_error(str(md_file), e)
                continue
        
        # Save all chunks
        all_chunks_file = output_dir / "all_chunks.json"
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.logger.info(f"All chunks saved to: {all_chunks_file}")
        
        # Final summary
        if all_chunks:
            all_token_counts = [c['metadata']['token_count'] for c in all_chunks]
            all_levels = [c['metadata']['level'] for c in all_chunks]
            
            level_counts = {}
            for lvl in all_levels:
                level_counts[lvl] = level_counts.get(lvl, 0) + 1
            
            min_size = chunking_config.get('min_chunk_size', 150)
            max_size = chunking_config.get('max_chunk_size', 800)
            
            logger.logger.info("\n" + "="*60)
            logger.logger.info("IMPROVED HIERARCHICAL CHUNKING - FINAL SUMMARY")
            logger.logger.info("="*60)
            logger.logger.info(f"Total chunks: {len(all_chunks)}")
            logger.logger.info(f"Level distribution: {level_counts}")
            logger.logger.info(f"Total tokens: {sum(all_token_counts):,}")
            logger.logger.info(f"Avg tokens/chunk: {sum(all_token_counts)/len(all_chunks):.1f}")
            logger.logger.info(f"Min tokens: {min(all_token_counts)}")
            logger.logger.info(f"Max tokens: {max(all_token_counts)}")
            
            small = sum(1 for t in all_token_counts if t < min_size)
            optimal = sum(1 for t in all_token_counts if min_size <= t <= max_size)
            large = sum(1 for t in all_token_counts if t > max_size)
            
            logger.logger.info(f"\nChunk Size Distribution:")
            logger.logger.info(f"  Small (<{min_size}): {small} ({small/len(all_chunks)*100:.1f}%)")
            logger.logger.info(f"  Optimal ({min_size}-{max_size}): {optimal} ({optimal/len(all_chunks)*100:.1f}%)")
            logger.logger.info(f"  Large (>{max_size}): {large} ({large/len(all_chunks)*100:.1f}%)")
            
            # Quality metrics
            very_small = sum(1 for t in all_token_counts if t < 50)
            logger.logger.info(f"\nQuality Metrics:")
            logger.logger.info(f"  Chunks with < 50 tokens: {very_small} ({very_small/len(all_chunks)*100:.1f}%)")
            logger.logger.info(f"  Chunks in ideal range (200-600): {sum(1 for t in all_token_counts if 200 <= t <= 600)}")
            logger.logger.info("="*60)
        
    finally:
        logger.finalize()


if __name__ == "__main__":
    process_all_files()