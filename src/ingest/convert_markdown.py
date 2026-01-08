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
        for page in reader.pages:
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
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            md_text += text + "\n\n"
                
                # Save with .md extension
                output_file = output_dir / f"{pdf_path.stem}.md"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(md_text)
                
                print(f"Converted: {pdf_path} -> {output_file}")
            
            except Exception as e:
                print(f"Error converting {pdf_path}: {e}")
