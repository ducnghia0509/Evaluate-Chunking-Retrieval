# file này để tạo các câu hỏi dựa trên mẫu được gen từ deepseek, copy vào file "fix"

import json
import re
from typing import List, Dict, Any
from find_chunk import find_chunks_with_context_preprocessed

def parse_fix_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse file fix để lấy các câu hỏi và ngữ cảnh theo từng category
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split theo category
    sections = re.split(r'={50,}\n([A-Z\s]+)\n={50,}', content)
    
    results = []
    current_category = None
    
    for i in range(1, len(sections), 2):
        category_name = sections[i].strip()
        category_content = sections[i+1] if i+1 < len(sections) else ""
        
        # Map category name
        if "DOC" in category_name and "LAW" not in category_name:
            current_category = "doc"
        elif "DOCUMENT" in category_name and "LAW" in category_name:
            current_category = "document_law"
        elif "FICTION" in category_name:
            current_category = "fiction"
        else:
            continue
        
        # Parse questions trong category này
        questions = parse_questions(category_content, current_category)
        results.extend(questions)
    
    return results

def parse_questions(content: str, category: str) -> List[Dict[str, Any]]:
    """
    Parse các câu hỏi và ngữ cảnh từ content của một category
    Đọc từng dòng để xử lý chính xác câu hỏi có chứa dấu ngoặc kép
    """
    questions = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Tìm dòng bắt đầu bằng "Câu hỏi:"
        if line.startswith('Câu hỏi:'):
            # Lấy phần sau "Câu hỏi:", loại bỏ dấu " đầu và cuối
            question_part = line[len('Câu hỏi:'):].strip()
            
            # Tìm dấu " đầu tiên và cuối cùng
            if question_part.startswith('"') and question_part.endswith('"'):
                question_text = question_part[1:-1].strip()
            else:
                # Trường hợp không có dấu " đầu/cuối hoặc format khác
                i += 1
                continue
            
            i += 1
            
            # Tìm dòng "Ngữ cảnh liên quan:"
            contexts = []
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith('Ngữ cảnh liên quan:'):
                    # Kiểm tra xem ngay sau "Ngữ cảnh liên quan:" có nội dung không
                    context_part = line[len('Ngữ cảnh liên quan:'):].strip()
                    
                    if context_part.startswith('"') and context_part.endswith('"'):
                        # Context nằm ngay trên cùng dòng
                        contexts.append(context_part[1:-1].strip())
                    
                    i += 1
                    
                    # Đọc các dòng tiếp theo cho đến khi gặp dòng trống hoặc "Câu hỏi:" mới
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        # Nếu gặp dòng trống hoặc "Câu hỏi:" mới hoặc "===" => kết thúc
                        if not line or line.startswith('Câu hỏi:') or line.startswith('==='):
                            break
                        
                        # Nếu dòng bắt đầu bằng ", đây là một ngữ cảnh
                        if line.startswith('"') and line.endswith('"'):
                            contexts.append(line[1:-1].strip())
                        
                        i += 1
                    
                    break
                
                i += 1
            
            # Thêm câu hỏi vào danh sách nếu có ngữ cảnh
            if contexts:
                questions.append({
                    "question": question_text,
                    "contexts": contexts,
                    "category": category
                })
        else:
            i += 1
    
    return questions

def find_chunks_for_contexts(types: str,contexts: List[str], category: str) -> List[Dict[str, Any]]:
    """
    Tìm chunk phù hợp cho MỖI đoạn context riêng biệt
    Trả về danh sách các chunk_id duy nhất
    """
    # Xác định file chunks dựa trên category
    simple = "/all_chunks"
    if types == "parent_child":
        simple = "/all_child_chunks"
    chunk_file_map = {
        "doc": "Chunked/"+types+ simple + ".json",
        "document_law": "Chunked/"+types+ simple + ".json",
        "fiction": "Chunked/"+types+ simple + ".json"
    }
    chunk_file_path = "Chunked/"+types+ simple + ".json"
    chunk_file = chunk_file_map.get(category, chunk_file_path)

    # Dictionary để lưu chunks duy nhất và match_percentage cao nhất
    found_chunks = {}
    
    # Tìm chunks cho từng đoạn context
    for ctx_idx, context in enumerate(contexts, 1):
        print(f"      - Đoạn ngữ cảnh {ctx_idx}/{len(contexts)} ({len(context)} ký tự)...")
        
        # Tìm NHIỀU chunks (top_k=5) thay vì chỉ 1
        results = find_chunks_with_context_preprocessed(chunk_file, context, top_k=3)
        
        if results and len(results) > 0:
            # Duyệt qua TẤT CẢ kết quả tìm được
            for result in results:
                chunks = result.get('chunks', [])
                match_percentage = result.get('score_info', {}).get('match_percentage', 0)
                
                # Chỉ lấy chunks có match >= 70%
                if match_percentage >= 70:
                    for chunk in chunks:
                        chunk_id = chunk.get('chunk_id', '')
                        
                        # Nếu chunk_id chưa có hoặc có match_percentage cao hơn, cập nhật
                        if chunk_id not in found_chunks or found_chunks[chunk_id]['match_percentage'] < match_percentage:
                            found_chunks[chunk_id] = {
                                'match_percentage': match_percentage
                            }
                        
                        print(f"         ✓ Tìm thấy: {chunk_id} (match: {match_percentage:.1f}%)")
    
    # Chuyển sang format output
    result_chunks = []
    for chunk_id, info in found_chunks.items():
        match_percentage = info['match_percentage']
        
        # Xác định relevance dựa trên match percentage
        if match_percentage >= 90:
            relevance = "high"
        elif match_percentage >= 70:
            relevance = "medium"
        else:
            relevance = "low"
        
        result_chunks.append({
            "chunk_id": chunk_id,
            "relevance": relevance
        })
    
    return result_chunks

def create_evaluation_dataset(types):
    """
    Tạo file evaluation dataset
    """
    print("🔍 Đang parse file fix...")
    questions = parse_fix_file("fix")
    
    print(f"✅ Đã parse {len(questions)} câu hỏi")
    
    evaluation_data = []
    
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] Đang xử lý câu hỏi: {q['question'][:50]}...")
        print(f"   Category: {q['category']}")
        print(f"   Số đoạn ngữ cảnh: {len(q['contexts'])}")
        
        # Tìm chunks cho TẤT CẢ các đoạn contexts
        found_chunks = find_chunks_for_contexts(types, q['contexts'], q['category'])
        
        if found_chunks:
            print(f"   ✅ Tìm thấy {len(found_chunks)} chunk(s)")
            
            evaluation_data.append({
                "id": f"{q['category']}_{idx}",
                "category": q['category'],
                "question": q['question'],
                "query": q['question'],  # Giữ nguyên như yêu cầu
                "relevant_chunks": found_chunks
            })
        else:
            print(f"   ❌ Không tìm thấy chunk phù hợp")
            # Vẫn thêm vào nhưng không có chunk
            evaluation_data.append({
                "id": f"{q['category']}_{idx}",
                "category": q['category'],
                "question": q['question'],
                "query": q['question'],
                "relevant_chunks": []
            })
    
    # Lưu file JSON
    output_file = types + "_evaluation_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ ĐÃ TẠO FILE EVALUATION DATASET")
    print(f"{'='*80}")
    print(f"📁 File: {output_file}")
    print(f"📊 Tổng số câu hỏi: {len(evaluation_data)}")
    
    # Thống kê
    categories = {}
    with_chunks = 0
    without_chunks = 0
    multi_chunks = 0  # Số câu hỏi có nhiều hơn 1 chunk
    
    for item in evaluation_data:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
        
        num_chunks = len(item['relevant_chunks'])
        if num_chunks > 0:
            with_chunks += 1
            if num_chunks > 1:
                multi_chunks += 1
        else:
            without_chunks += 1
    
    print(f"\n📈 THỐNG KÊ:")
    print(f"   • Câu hỏi có chunk: {with_chunks}")
    print(f"   • Câu hỏi có NHIỀU HƠNS 1 chunk: {multi_chunks}")
    print(f"   • Câu hỏi không có chunk: {without_chunks}")
    print(f"\n   Theo category:")
    for cat, count in categories.items():
        print(f"   • {cat}: {count} câu hỏi")
    
    return evaluation_data

if __name__ == "__main__":
    # typess = ["fixed", "hierarchical", "semantic", "structure_paragraph"]
    typess = ["parent_child"]
    for types in typess:
        print(f"=============BEGIN {types}=============")
        create_evaluation_dataset(types)
        print(f"=============END {types}=============")

