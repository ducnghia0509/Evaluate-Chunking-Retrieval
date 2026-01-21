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

def split_by_sentences(text: str, max_size: int, min_size: int) -> List[str]:
    """
    Split text by sentences with size constraints
    """
    # Improved sentence splitting regex
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    result = []
    current_chunk = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = count_tokens(sent)
        
        if current_tokens + sent_tokens > max_size:
            if current_chunk:
                result.append(' '.join(current_chunk))
            
            # Start new chunk
            if sent_tokens > max_size:
                # Sentence itself is too large, split by words
                words = sent.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    word_token = count_tokens(word)
                    if word_tokens + word_token <= max_size:
                        word_chunk.append(word)
                        word_tokens += word_token
                    else:
                        if word_chunk:
                            result.append(' '.join(word_chunk))
                        word_chunk = [word]
                        word_tokens = word_token
                
                if word_chunk:
                    result.append(' '.join(word_chunk))
            else:
                current_chunk = [sent]
                current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens
    
    if current_chunk:
        result.append(' '.join(current_chunk))
    
    return result

def smart_split_paragraphs(text: str, max_size: int, min_size: int) -> List[str]:
    """
    Smart paragraph splitting with size constraints
    """
    # First, split by actual paragraphs (blank lines)
    paragraphs = []
    current_para = []
    
    for line in text.split('\n'):
        if line.strip() == '':
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
        else:
            current_para.append(line)
    
    if current_para:
        paragraphs.append('\n'.join(current_para))
    
    if not paragraphs:
        return [text]
    
    # Smart merging and splitting
    result = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        # If paragraph itself is too large
        if para_tokens > max_size:
            # Save current chunk if exists
            if current_chunk:
                result.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split the large paragraph
            sub_chunks = split_by_sentences(para, max_size, min_size)
            result.extend(sub_chunks)
        elif current_tokens + para_tokens <= max_size:
            # Can add to current chunk
            current_chunk.append(para)
            current_tokens += para_tokens
        else:
            # Current chunk is full
            if current_chunk:
                # Check if chunk is too small
                if current_tokens < min_size and result:
                    # Try to merge with last chunk
                    last_chunk = result[-1]
                    last_tokens = count_tokens(last_chunk)
                    
                    if last_tokens + current_tokens <= max_size:
                        result[-1] = last_chunk + '\n\n' + '\n\n'.join(current_chunk)
                    else:
                        result.append('\n\n'.join(current_chunk))
                else:
                    result.append('\n\n'.join(current_chunk))
            
            # Start new chunk
            current_chunk = [para]
            current_tokens = para_tokens
    
    # Handle last chunk
    if current_chunk:
        if current_tokens < min_size and result:
            # Try to merge with last chunk
            last_chunk = result[-1]
            last_tokens = count_tokens(last_chunk)
            
            if last_tokens + current_tokens <= max_size:
                result[-1] = last_chunk + '\n\n' + '\n\n'.join(current_chunk)
            else:
                result.append('\n\n'.join(current_chunk))
        else:
            result.append('\n\n'.join(current_chunk))
    
    return result

def extract_markdown_structure(text: str) -> List[Tuple[str, str, int]]:
    """
    Extract markdown structure with headers and their content
    Returns: [(header, content, level)]
    """
    lines = text.split('\n')
    sections = []
    current_header = None
    current_level = 0
    current_content = []
    
    for line in lines:
        # Check for headers
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
    
    # Add last section
    if current_header is not None:
        sections.append((current_header, '\n'.join(current_content).strip(), current_level))
    elif current_content:
        # No headers, treat as one section
        sections.append(("Document", '\n'.join(current_content).strip(), 0))
    
    return sections

def extract_complete_sections(text: str) -> List[Tuple[str, str, str, int]]:
    """
    Extract complete markdown sections with headers included in content
    Returns: [(full_header_line, header_text, full_content, level)]
    """
    lines = text.split('\n')
    sections = []
    
    current_full_header = None  # Full header line với dấu #
    current_header_text = None  # Chỉ nội dung header
    current_level = 0
    current_content_lines = []
    
    for line in lines:
        # Check for headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            # Save previous section if exists
            if current_full_header is not None:
                # Tạo full content bao gồm cả header
                full_content_lines = [current_full_header] + current_content_lines
                full_content = '\n'.join(full_content_lines).strip()
                
                sections.append((
                    current_full_header,      # Full header line: "# ## Header"
                    current_header_text,      # Chỉ text: "Header"
                    full_content,             # Toàn bộ section bao gồm header
                    current_level
                ))
            
            # Start new section
            current_level = len(header_match.group(1))
            current_full_header = line  # Giữ nguyên cả dấu #
            current_header_text = header_match.group(2)
            current_content_lines = []
        else:
            current_content_lines.append(line)
    
    # Add last section
    if current_full_header is not None:
        full_content_lines = [current_full_header] + current_content_lines
        full_content = '\n'.join(full_content_lines).strip()
        sections.append((current_full_header, current_header_text, full_content, current_level))
    elif current_content_lines:
        # No headers in document, treat as one section
        full_content = '\n'.join(current_content_lines).strip()
        sections.append(("# Document", "Document", full_content, 0))
    
    return sections

def create_chunk_with_header(header: str, content: str, level: int, 
                           chunk_text: str, chunk_idx: int, 
                           file_stem: str, file_name: str,
                           max_size: int, min_size: int,
                           para_idx: int, total_paras: int) -> Dict:
    """
    Create a chunk with proper header inclusion
    """
    # QUAN TRỌNG: Đảm bảo chunk text có chứa header nếu cần và không vượt max_size
    final_text = chunk_text
    header_included = False
    
    # Nếu chunk là paragraph đầu tiên trong section và không bắt đầu bằng header
    if para_idx == 0 and not chunk_text.strip().startswith('#'):
        # Thêm header vào chunk text
        # Tìm level từ header gốc
        header_level = level
        header_prefix = '#' * header_level + ' ' if header_level > 0 else ''
        header_with_prefix = f"{header_prefix}{header}"
        
        # Kiểm tra xem thêm header có vượt max_size không
        combined_text = f"{header_with_prefix}\n\n{chunk_text}"
        combined_tokens = count_tokens(combined_text)
        
        if combined_tokens <= max_size:
            final_text = combined_text
            header_included = True
        # Nếu vượt, cắt bớt chunk_text để fit với header
        elif count_tokens(header_with_prefix) < max_size * 0.3:  # Header không quá 30% max_size
            # Cắt chunk_text để fit
            available_tokens = max_size - count_tokens(header_with_prefix) - 10  # Reserve 10 tokens
            if available_tokens > 50:
                # Lấy phần đầu của chunk_text
                words = chunk_text.split()
                truncated = []
                temp_text = header_with_prefix + '\n\n'
                
                for word in words:
                    test_text = temp_text + ' '.join(truncated + [word])
                    if count_tokens(test_text) <= max_size:
                        truncated.append(word)
                    else:
                        break
                
                if truncated:
                    final_text = header_with_prefix + '\n\n' + ' '.join(truncated)
                    header_included = True
    
    token_count = count_tokens(final_text)
    
    return {
        'chunk_id': f"{file_stem}_chunk_{chunk_idx}",
        'source_file': file_name,
        'chunk_index': chunk_idx,
        'text': final_text.strip(),
        'metadata': {
            'strategy': 'structure_paragraph',
            'section_header': header,
            'header_level': level,
            'full_header_line': '#' * level + ' ' + header if level > 0 else header,
            'paragraph_index': para_idx,
            'total_paragraphs': total_paras,
            'token_count': token_count,
            'has_header_in_text': final_text.strip().startswith('#'),
            'chunk_info': {
                'max_size': max_size,
                'min_size': min_size,
                'size_ratio': token_count / max_size if max_size > 0 else 0,
                'header_included': final_text != chunk_text
            }
        }
    }


def smart_split_section_with_header(full_content: str, header: str, level: int,
                                  max_size: int, min_size: int) -> List[str]:
    """
    Split a section into chunks while preserving header context
    """
    # Tách lines
    lines = full_content.split('\n')
    
    # Nếu toàn bộ section nhỏ hơn max_size, giữ nguyên
    if count_tokens(full_content) <= max_size:
        return [full_content]
    
    # Phân tích cấu trúc
    paragraphs = []
    current_paragraph = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Dòng trống đánh dấu kết thúc paragraph
        if not line_stripped:
            if current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = []
        # Header line - bắt đầu paragraph mới
        elif line_stripped.startswith('#'):
            if current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))
            current_paragraph = [line]
        else:
            current_paragraph.append(line)
    
    # Thêm paragraph cuối
    if current_paragraph:
        paragraphs.append('\n'.join(current_paragraph))
    
    # Merge và split paragraphs
    chunks = []
    current_chunk_lines = []
    current_tokens = 0
    
    for i, para in enumerate(paragraphs):
        para_tokens = count_tokens(para)
        
        # Nếu paragraph riêng lẻ quá lớn
        if para_tokens > max_size:
            # Lưu chunk hiện tại nếu có
            if current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = []
                current_tokens = 0
            
            # Chia paragraph lớn
            if para.startswith('#'):
                # Header paragraph, giữ nguyên
                chunks.append(para)
            else:
                # Content paragraph, chia nhỏ
                sub_chunks = split_large_paragraph(para, max_size)
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
    
    # Post-process: Đảm bảo chunks không quá nhỏ và không vượt max_size
    merged_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk)
        
        # Nếu chunk vượt quá max_size, cần split thêm
        if chunk_tokens > max_size:
            # Split chunk lớn này
            sub_chunks = split_large_paragraph(chunk, max_size)
            for sub_chunk in sub_chunks:
                sub_tokens = count_tokens(sub_chunk)
                # Merge với chunk trước nếu quá nhỏ
                if sub_tokens < 50 and merged_chunks:
                    merged_chunks[-1] = merged_chunks[-1] + '\n\n' + sub_chunk
                else:
                    merged_chunks.append(sub_chunk)
        # Nếu chunk quá nhỏ, merge với chunk trước
        elif chunk_tokens < 50 and merged_chunks:
            # Kiểm tra xem merge có vượt max_size không
            prev_tokens = count_tokens(merged_chunks[-1])
            if prev_tokens + chunk_tokens <= max_size:
                merged_chunks[-1] = merged_chunks[-1] + '\n\n' + chunk
            else:
                merged_chunks.append(chunk)
        else:
            merged_chunks.append(chunk)
    
    return merged_chunks


def split_large_paragraph(paragraph: str, max_size: int) -> List[str]:
    """Split large paragraph by sentences"""
    # Ưu tiên split bằng dấu xuống dòng trước
    if '\n' in paragraph:
        lines = paragraph.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens <= max_size:
                current_chunk.append(line)
                current_tokens += line_tokens
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    # Split bằng sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = count_tokens(sent)
        
        if current_tokens + sent_tokens <= max_size:
            current_chunk.append(sent)
            current_tokens += sent_tokens
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            current_tokens = sent_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def structure_paragraph_chunking(markdown_file: Path, config: Dict) -> List[Dict]:
    """
    Fixed structure-aware paragraph chunking with header preservation
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    max_chunk_size = config.get('max_chunk_size', 600)
    min_chunk_size = config.get('min_chunk_size', 100)
    
    # Extract sections với header được bảo toàn
    sections = extract_complete_sections(content)
    
    chunks = []
    chunk_idx = 0
    
    for full_header_line, header_text, full_content, level in sections:
        if not full_content.strip():
            continue
        
        # Split section thành các chunks
        text_chunks = smart_split_section_with_header(
            full_content, header_text, level, max_chunk_size, min_chunk_size
        )
        
        # Tạo chunk objects
        for chunk_idx_in_section, chunk_text in enumerate(text_chunks):
            chunk = create_chunk_with_header(
                header=header_text,
                content=full_content,
                level=level,
                chunk_text=chunk_text,
                chunk_idx=chunk_idx,
                file_stem=markdown_file.stem,
                file_name=markdown_file.name,
                max_size=max_chunk_size,
                min_size=min_chunk_size,
                para_idx=chunk_idx_in_section,
                total_paras=len(text_chunks)
            )
            
            chunks.append(chunk)
            chunk_idx += 1
    
    # Post-processing: Merge chunks quá nhỏ (<50 tokens) vào chunk trước
    min_token_threshold = 50
    merged_chunks = []
    
    for i, chunk in enumerate(chunks):
        token_count = chunk['metadata']['token_count']
        
        # Nếu chunk quá nhỏ và có chunk trước đó
        if token_count < min_token_threshold and merged_chunks:
            prev_chunk = merged_chunks[-1]
            
            # Kiểm tra xem merge có vượt max_chunk_size không
            combined_text = prev_chunk['text'] + '\n\n' + chunk['text']
            combined_tokens = count_tokens(combined_text)
            
            if combined_tokens <= max_chunk_size:
                # Merge vào chunk trước
                prev_chunk['text'] = combined_text
                prev_chunk['metadata']['token_count'] = combined_tokens
                prev_chunk['metadata']['merged_count'] = prev_chunk['metadata'].get('merged_count', 0) + 1
            else:
                # Không merge được, giữ riêng
                merged_chunks.append(chunk)
        else:
            merged_chunks.append(chunk)
    
    # Re-index chunks sau khi merge
    for idx, chunk in enumerate(merged_chunks):
        chunk['chunk_id'] = f"{markdown_file.stem}_chunk_{idx}"
        chunk['chunk_index'] = idx
    
    return merged_chunks



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
