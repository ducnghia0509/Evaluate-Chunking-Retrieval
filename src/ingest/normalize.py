import os
import re
from pathlib import Path


markdown_folder = Path("../../Markdown")


def remove_empty_lines(text):
    """Remove consecutive empty lines, keep only single newlines"""
    # Replace multiple newlines with double newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text


def remove_meaningless_hashes(text):
    """Remove meaningless ### patterns like '###:'"""
    text = re.sub(r'^###:\s*$', '', text, flags=re.MULTILINE)
    return text


def remove_page_markers(text):
    """Remove page markers like '--- Page 1 ---', '--- Page 2 ---', etc."""
    text = re.sub(r'^---\s*Page\s+\d+\s*---\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    return text


def remove_duplicate_header_content(text):
    """
    Remove duplicate content after headers
    Example:
    # DẪN NHẬP
    DẪN NHẬP
    
    becomes:
    # DẪN NHẬP
    """
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a header line
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            result.append(line)
            header_text = header_match.group(2).strip()
            
            # Check next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                result.append(lines[j])
                j += 1
            
            if j < len(lines):
                next_line = lines[j].strip()
                
                # Pattern 1: Exact match
                if next_line == header_text:
                    # Skip the duplicate line
                    i = j + 1
                    continue
                
                # Pattern 2: Header has number prefix like "13. Title" and next line has "13\nTitle"
                # Extract potential number prefix
                number_match = re.match(r'^(\d+)[\.\s]+(.+)$', header_text)
                if number_match:
                    number = number_match.group(1)
                    title = number_match.group(2).strip()
                    
                    # Check if next line is just the number
                    if next_line == number:
                        # Check line after that
                        k = j + 1
                        while k < len(lines) and not lines[k].strip():
                            k += 1
                        
                        if k < len(lines) and lines[k].strip() == title:
                            # Skip both number and title lines
                            i = k + 1
                            continue
                    # Check if next line is the title without number
                    elif next_line == title:
                        i = j + 1
                        continue
            
            i += 1
        else:
            result.append(line)
            i += 1
    
    return '\n'.join(result)


def normalize_markdown_file(file_path):
    """Normalize a single markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply normalizations
    content = remove_page_markers(content)
    content = remove_empty_lines(content)
    content = remove_meaningless_hashes(content)
    
    # Special handling for sherlock holmes file
    if "sherlock-holmes-toan-tap" in file_path.name:
        content = remove_duplicate_header_content(content)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Normalized: {file_path}")


def merge_land_law_files():
    """Merge 31-2024-qh15_*.md files into single 31-2024-qh15.md"""
    doc_law_folder = markdown_folder / "document_law"
    
    # Find all parts
    parts = sorted(doc_law_folder.glob("31-2024-qh15_*.md"))
    
    if not parts:
        print("No land law files found to merge")
        return
    
    # Merge content
    merged_content = ""
    for part_file in parts:
        with open(part_file, 'r', encoding='utf-8') as f:
            content = f.read()
            merged_content += content + "\n\n"
    
    # Normalize merged content
    merged_content = remove_page_markers(merged_content)
    merged_content = remove_empty_lines(merged_content)
    merged_content = remove_meaningless_hashes(merged_content)
    
    # Save to final file
    output_file = doc_law_folder / "31-2024-qh15.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    print(f"Merged {len(parts)} files into: {output_file}")
    
    # Delete old part files
    for part_file in parts:
        part_file.unlink()
        print(f"Deleted: {part_file}")


# Main processing
print("Starting normalization process...\n")

# First, merge land law files
merge_land_law_files()
print()

# Process all markdown files
for root, dirs, files in os.walk(markdown_folder):
    for file in files:
        if file.lower().endswith('.md'):
            file_path = Path(root) / file
            normalize_markdown_file(file_path)

print("\nNormalization complete!")
