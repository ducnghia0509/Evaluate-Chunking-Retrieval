import os
import re
from pathlib import Path
from pypdf import PdfReader


source_folder = Path("../../Sources")
markdown_folder = Path("../../Markdown")


def get_outline_headers(outline, reader, level=0, has_numbering=None):
    """
    Extract outline information with page numbers
    Returns list of (page_num, header_text) tuples
    """
    headers = []
    
    # First pass: check if outline has any numbering
    if has_numbering is None:
        has_numbering = any(
            re.match(r'^\d+\.', item.title) if not isinstance(item, list) else False
            for item in outline
        )
    
    for item in outline:
        if isinstance(item, list):
            headers.extend(get_outline_headers(item, reader, level + 1, has_numbering))
        else:
            title = item.title.strip()
            
            # Get page number for this outline item
            try:
                page_num = reader.get_destination_page_number(item)
            except:
                page_num = None
            
            # Determine header level
            header_text = ""
            if re.match(r'^Abstract', title, re.IGNORECASE):
                header_text = f"# {title}"
            elif has_numbering:
                if re.match(r'^\d+\.\d+', title):
                    header_text = f"## {title}"
                elif re.match(r'^\d+\.', title):
                    header_text = f"# {title}"
                else:
                    prefix = "#" * min(level + 1, 6) if level > 0 else "#"
                    header_text = f"{prefix} {title}"
            else:
                header_text = f"# {title}"
            
            if page_num is not None:
                headers.append((page_num, header_text))
    
    return headers


def analyze_table_of_contents(md_text):
    """
    Analyze the structure of table of contents in the first 5 pages
    Returns dict with structure information or None if no TOC found
    """
    # Extract first 5 pages
    pages = md_text.split('--- Page ')
    first_5_pages = ''.join(pages[:6])  # 0 is empty, 1-5 are first 5 pages
    
    # Check if has table of contents or introduction
    if not (re.search(r'MỤC LỤC', first_5_pages, re.IGNORECASE) or 
            re.search(r'Lời giới thiệu', first_5_pages, re.IGNORECASE)or 
            re.search(r'NỘI DUNG', first_5_pages, re.IGNORECASE)):
        return None
    
    structure = {
        'has_toc': True,
        'main_level': None,  # e.g., 'PHẦN', 'Chương'
        'main_numbering': None,  # 'roman' or 'arabic'
        'main_pattern': None,  # regex pattern for main level
        'sub_patterns': []  # list of regex patterns for sub levels
    }
    
    # Detect main level patterns (ordered by priority - larger to smaller)
    main_patterns = [
        (r'^PHẦN\s+([IVXLCDM]+)', 'PHẦN', 'roman'),
        (r'^Phần\s+([IVXLCDM]+)', 'Phần', 'roman'),
        (r'^PHẦN\s+(\d+)', 'PHẦN', 'arabic'),
        (r'^Phần\s+(\d+)', 'Phần', 'arabic'),
        (r'^CHƯƠNG\s+([IVXLCDM]+)', 'CHƯƠNG', 'roman'),
        (r'^Chương\s+([IVXLCDM]+)', 'Chương', 'roman'),
        (r'^CHƯƠNG\s+(\d+)', 'CHƯƠNG', 'arabic'),
        (r'^Chương\s+(\d+)', 'Chương', 'arabic'),
        (r'^([IVXLCDM]+)\.', 'Roman', 'roman'),
        (r'^(\d+)\.', 'Number', 'arabic'),
    ]
    
    lines = first_5_pages.split('\n')
    
    # Find main level
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for pattern, level_name, numbering_type in main_patterns:
            if re.match(pattern, line):
                if structure['main_level'] is None:
                    structure['main_level'] = level_name
                    structure['main_numbering'] = numbering_type
                    structure['main_pattern'] = pattern
                    break
        if structure['main_level']:
            break
    
    # If no main pattern found, check for uppercase headings
    if structure['main_level'] is None:
        # Find lines that are all uppercase and likely to be headings
        uppercase_candidates = []
        in_toc = False
        for line in lines:
            stripped = line.strip()
            if re.search(r'MỤC LỤC|NỘI DUNG|Mục lục|Nội dung', stripped, re.IGNORECASE):
                in_toc = True
                continue
            if in_toc and stripped and len(stripped) > 5:
                # Check if line is mostly uppercase letters (excluding dots and spaces)
                letters_only = re.sub(r'[^a-zA-ZÀ-ỹ]', '', stripped)
                if letters_only and letters_only.isupper() and len(letters_only) >= 3:
                    uppercase_candidates.append(stripped)
                    if len(uppercase_candidates) >= 3:  # Found at least 3 uppercase headings
                        structure['main_level'] = 'UPPERCASE'
                        structure['main_numbering'] = 'none'
                        structure['main_pattern'] = r'^[A-ZÀ-Ỹ][A-ZÀ-Ỹ\s]+[A-ZÀ-Ỹ]$'
                        break
    
    # Detect sub-level patterns
    sub_pattern_candidates = [
        (r'^\d+\.\d+', '1.1, 2.3 format'),  # 1.1, 2.3
        (r'^[IVXLCDM]+\.\d+', 'I.1, II.2 format'),  # I.1, II.2
        (r'^\d{2,}:', '01:, 05: format'),  # 01:, 05:
        (r'^Thủ tục\s+\d+:', 'Thủ tục XX: format'),
        (r'^Phụ lục\s+\d+', 'Phụ lục XX format'),
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for sub_pattern, description in sub_pattern_candidates:
            match = re.match(sub_pattern, line)
            if match and sub_pattern not in [p[0] for p in structure['sub_patterns']]:
                structure['sub_patterns'].append((sub_pattern, description))
    
    return structure


def apply_toc_structure_to_markdown(md_text, toc_structure):
    """
    Apply detected TOC structure to markdown text
    Add heading tags based on detected patterns
    """
    if not toc_structure or not toc_structure['main_pattern']:
        return md_text
    
    lines = md_text.split('\n')
    new_lines = []
    i = 0
    current_main_number = None  # Track current main section number
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and already formatted headers
        if not stripped or stripped.startswith('#'):
            new_lines.append(line)
            i += 1
            continue
        
        # Check main level pattern
        main_match = re.match(toc_structure['main_pattern'], stripped)
        if main_match:
            # For patterns like "PHẦN I", "Chương 1", etc.
            if toc_structure['main_level'] in ['PHẦN', 'Phần', 'CHƯƠNG', 'Chương']:
                # Extract number/roman for tracking
                number_part = main_match.group(1)
                current_main_number = number_part
                
                new_lines.append(f"# {stripped}")
                i += 1
                # Check if next line should also be heading (e.g., uppercase title)
                if i < len(lines):
                    next_stripped = lines[i].strip()
                    if next_stripped and next_stripped.isupper() and len(next_stripped) > 3:
                        new_lines.append(f"# {next_stripped}")
                        i += 1
                continue
            # For uppercase headings
            elif toc_structure['main_level'] == 'UPPERCASE':
                # Check if line is mostly uppercase
                letters_only = re.sub(r'[^a-zA-ZÀ-ỹ]', '', stripped)
                if letters_only and letters_only.isupper() and len(letters_only) >= 3:
                    new_lines.append(f"# {stripped}")
                    i += 1
                    continue
            # For simple patterns like "I.", "1.", "2.", "3."
            elif toc_structure['main_level'] in ['Roman', 'Number']:
                # Extract the number/roman part
                number_part = main_match.group(1)
                current_main_number = number_part
                
                new_lines.append(f"# {stripped}")
                i += 1
                continue
            else:
                new_lines.append(f"# {stripped}")
                i += 1
                continue
        
        # Check sub-level patterns with parent relationship
        sub_matched = False
        for sub_pattern, _ in toc_structure['sub_patterns']:
            sub_match = re.match(sub_pattern, stripped)
            if sub_match:
                # For patterns like "3.1", "3.2" - check if parent is "3"
                if re.match(r'^\d+\.\d+', stripped):
                    parent_num = stripped.split('.')[0]
                    # Only mark as sub-heading if it matches current main number
                    if current_main_number and parent_num == current_main_number:
                        new_lines.append(f"## {stripped}")
                        sub_matched = True
                        break
                    # If no current main number or doesn't match, treat as potential main
                    elif not current_main_number:
                        new_lines.append(f"## {stripped}")
                        sub_matched = True
                        break
                # For patterns like "I.1", "II.2" - check if parent is "I", "II"
                elif re.match(r'^[IVXLCDM]+\.\d+', stripped):
                    parts = stripped.split('.')
                    parent_roman = parts[0]
                    if current_main_number and parent_roman == current_main_number:
                        new_lines.append(f"## {stripped}")
                        sub_matched = True
                        break
                    elif not current_main_number:
                        new_lines.append(f"## {stripped}")
                        sub_matched = True
                        break
                # For other patterns, just apply sub-heading
                else:
                    new_lines.append(f"## {stripped}")
                    sub_matched = True
                    break
        
        if sub_matched:
            i += 1
            continue
        
        # No match, keep original line
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)


def remove_table_of_contents(md_text, toc_structure):
    """
    Remove table of contents section from markdown
    Find TOC start, detect first main item, find its second occurrence, and remove everything in between
    """
    if not toc_structure or not toc_structure['main_pattern']:
        return md_text
    
    lines = md_text.split('\n')
    
    # Find TOC start position
    toc_start_idx = None
    for i, line in enumerate(lines):
        if re.search(r'MỤC LỤC|Mục lục|MỤC\s*LỤC|NỘI DUNG|Nội dung', line.strip(), re.IGNORECASE):
            toc_start_idx = i
            break
    
    if toc_start_idx is None:
        return md_text
    
    # Find first main item in TOC (after TOC marker)
    first_main_item = None
    first_main_item_idx = None
    
    for i in range(toc_start_idx + 1, min(toc_start_idx + 200, len(lines))):  # Search within next 200 lines
        stripped = lines[i].strip()
        if not stripped:
            continue
        
        # Check if matches main pattern
        main_match = re.match(toc_structure['main_pattern'], stripped)
        if main_match:
            first_main_item = stripped
            first_main_item_idx = i
            break
    
    if not first_main_item:
        return md_text
    
    # Find second occurrence of the first main item (actual content start)
    second_occurrence_idx = None
    
    for i in range(first_main_item_idx + 1, len(lines)):
        stripped = lines[i].strip()
        
        # Match the same pattern with same number/roman
        if toc_structure['main_level'] in ['PHẦN', 'Phần', 'CHƯƠNG', 'Chương']:
            # Extract the number/roman from first item
            match = re.match(toc_structure['main_pattern'], first_main_item)
            if match:
                number_part = match.group(1)
                # Check if current line has same pattern with same number
                current_match = re.match(toc_structure['main_pattern'], stripped)
                if current_match and current_match.group(1) == number_part:
                    second_occurrence_idx = i
                    break
        elif toc_structure['main_level'] == 'UPPERCASE':
            # For uppercase headings, match exact text or very similar
            if stripped == first_main_item:
                second_occurrence_idx = i
                break
        else:
            # For simple patterns like "I.", "1."
            if stripped == first_main_item or stripped.startswith(first_main_item.split()[0]):
                second_occurrence_idx = i
                break
    
    if not second_occurrence_idx:
        return md_text
    
    # Remove lines from TOC start to just before second occurrence
    # Keep everything before TOC and after second occurrence
    new_lines = lines[:toc_start_idx] + lines[second_occurrence_idx:]
    
    return '\n'.join(new_lines)


def convert_with_outline(reader):
    """
    Convert PDF with outline to markdown
    Extract all text and insert headers at appropriate positions
    """
    # Get outline headers with page numbers
    headers = get_outline_headers(reader.outline, reader)
    headers_dict = {}
    for page_num, header_text in headers:
        if page_num not in headers_dict:
            headers_dict[page_num] = []
        headers_dict[page_num].append(header_text)
    
    md_text = ""
    
    # Extract text page by page and insert headers
    for page_num, page in enumerate(reader.pages):
        # Add page marker
        md_text += f"--- Page {page_num + 1} ---\n\n"
        
        # Add headers for this page
        if page_num in headers_dict:
            for header in headers_dict[page_num]:
                md_text += f"{header}\n\n"
        
        # Extract page text
        text = page.extract_text()
        if text:
            md_text += text + "\n\n"
    
    return md_text


def convert_document_law_to_markdown(pdf_path):
    """
    Convert PDF from document_law folder to markdown
    - Add # when encountering "Chương" + Roman numeral + next line if all UPPERCASE
    - Add ## when encountering "Điều x. "
    """
    md_text = ""
    
    with PdfReader(pdf_path) as reader:
        for page_num, page in enumerate(reader.pages):
            # Add page marker
            md_text += f"--- Page {page_num + 1} ---\n\n"
            
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Check for "Chương" + Roman numeral
                chapter_match = re.match(r'^Chương\s+([IVXLCDM]+)', line, re.IGNORECASE)
                if chapter_match:
                    md_text += f"# {line}\n\n"
                    # Check next line if all uppercase
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and next_line.isupper():
                            md_text += f"# {next_line}\n\n"
                            i += 1  # Skip next line
                # Check for "Điều x. "
                elif re.match(r'^Điều\s+\d+\.', line, re.IGNORECASE):
                    md_text += f"## {line}\n\n"
                else:
                    md_text += f"{line}\n"
                
                i += 1
            
            md_text += "\n"
    
    return md_text


# Process all PDF files in Source folder
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_path = Path(root) / file
            relative_path = pdf_path.relative_to(source_folder)
            
            # Create corresponding directory structure in Markdown folder
            output_dir = markdown_folder / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            md_text = ""
            
            try:
                reader = PdfReader(pdf_path)
                
                # Check if PDF has outline
                if reader.outline:
                    md_text = convert_with_outline(reader)
                # If no outline and in document_law folder
                elif "document_law" in str(relative_path):
                    md_text = convert_document_law_to_markdown(pdf_path)
                else:
                    # Fallback: extract plain text
                    for page_num, page in enumerate(reader.pages):
                        # Add page marker
                        md_text += f"--- Page {page_num + 1} ---\n\n"
                        
                        text = page.extract_text()
                        if text:
                            md_text += text + "\n\n"
                
                # Analyze table of contents structure
                toc_structure = analyze_table_of_contents(md_text)
                if toc_structure:
                    print(f"  TOC Structure detected:")
                    print(f"    Main level: {toc_structure['main_level']} ({toc_structure['main_numbering']})")
                    if toc_structure['sub_patterns']:
                        print(f"    Sub patterns: {', '.join([desc for _, desc in toc_structure['sub_patterns']])}")
                    
                    # Apply TOC structure to markdown
                    md_text = apply_toc_structure_to_markdown(md_text, toc_structure)
                    print(f"  Applied TOC structure to markdown")
                    
                    # Remove table of contents section
                    original_length = len(md_text)
                    md_text = remove_table_of_contents(md_text, toc_structure)
                    if len(md_text) < original_length:
                        print(f"  Removed TOC section ({original_length - len(md_text)} characters)")
                
                # Save with .md extension
                output_file = output_dir / f"{pdf_path.stem}.md"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(md_text)
                
                print(f"Converted: {pdf_path} -> {output_file}")
            
            except Exception as e:
                print(f"Error converting {pdf_path}: {e}")
