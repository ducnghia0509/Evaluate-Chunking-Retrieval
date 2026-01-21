# tìm chunk bao hàm ngữ cảnh từ các câu hỏi
import json
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

file_find = "Chunked/fixed/all_chunks.json"
# context = "Hồ sơ đăng ký doanh nghiệp qua mạng thông tin điện tử phải được xác thực bằng chữ ký số hoặc Tài khoản đăng ký kinh doanh của người có thẩm quyền ký văn bản đề nghị đăng ký doanh nghiệp hoặc người được người có thẩm quyền ký văn bản đề nghị đăng ký doanh nghiệp ủy quyền thực hiện thủ tục đăng ký doanh nghiệp."

def preprocess_text_advanced(text: str) -> str:
    """
    Xử lý text nâng cao: xóa dấu, lowercase, chuẩn hóa
    """
    if not text:
        return ""
    
    # Chuyển về lowercase
    text = text.lower()
    
    # Xóa dấu tiếng Việt
    text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
    text = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', text)
    text = re.sub(r'[ìíịỉĩ]', 'i', text)
    text = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', text)
    text = re.sub(r'[ùúụủũưừứựửữ]', 'u', text)
    text = re.sub(r'[ỳýỵỷỹ]', 'y', text)
    text = re.sub(r'đ', 'd', text)
    
    # Xóa tất cả dấu câu và ký tự đặc biệt, chỉ giữ chữ và số
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def save_preprocessed_chunks(chunks: List[Dict], output_file: str):
    """
    Lưu các chunk đã được preprocess ra file mới
    """
    preprocessed_chunks = []
    
    for chunk in chunks:
        original_text = chunk.get('text', '')
        processed_text = preprocess_text_advanced(original_text)
        
        preprocessed_chunks.append({
            'original': chunk,
            'processed_text': processed_text,
            'processed_words': processed_text.split() if processed_text else [],
            'chunk_id': chunk.get('chunk_id', ''),
            'source_file': chunk.get('source_file', ''),
            'chunk_index': chunk.get('chunk_index', -1)
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu {len(preprocessed_chunks)} chunk đã xử lý vào {output_file}")
    return preprocessed_chunks

def find_exact_context_match(processed_chunks: List[Dict], context_processed: str, 
                           context_words: List[str]) -> List[Dict[str, Any]]:
    """
    Tìm chunk chứa toàn bộ context (chính xác hoặc gần chính xác)
    """
    results = []
    
    # Tìm exact match trong từng chunk
    for idx, chunk in enumerate(processed_chunks):
        chunk_text = chunk['processed_text']
        chunk_words = chunk['processed_words']
        
        # Kiểm tra xem context có nằm trong chunk không
        if context_processed in chunk_text:
            results.append({
                'type': 'exact_match_single',
                'chunks': [chunk['original']],
                'chunk_indices': [idx],
                'processed_chunk': chunk,
                'score_info': {
                    'match_type': 'exact',
                    'matched_words': len(context_words),
                    'total_context_words': len(context_words),
                    'match_percentage': 100.0,
                    'position': chunk_text.find(context_processed)
                }
            })
            continue
        
        # Tìm subsequence dài nhất
        if chunk_words:
            # Tìm subsequence dài nhất của context_words trong chunk_words
            max_match_length = 0
            max_match_start = 0
            
            # Sử dụng sliding window để tìm subsequence
            for i in range(len(chunk_words) - len(context_words) + 1):
                match_length = 0
                for j in range(len(context_words)):
                    if i + j < len(chunk_words) and chunk_words[i + j] == context_words[j]:
                        match_length += 1
                    else:
                        break
                
                if match_length > max_match_length:
                    max_match_length = match_length
                    max_match_start = i
            
            # Nếu tìm thấy match đáng kể (ít nhất 80% context)
            if max_match_length >= len(context_words) * 0.8:
                match_percentage = (max_match_length / len(context_words)) * 100
                
                results.append({
                    'type': 'subsequence_match_single',
                    'chunks': [chunk['original']],
                    'chunk_indices': [idx],
                    'processed_chunk': chunk,
                    'score_info': {
                        'match_type': 'subsequence',
                        'matched_words': max_match_length,
                        'total_context_words': len(context_words),
                        'match_percentage': match_percentage,
                        'position': max_match_start,
                        'matched_sequence': ' '.join(chunk_words[max_match_start:max_match_start + max_match_length])
                    }
                })
    
    return results

def find_continuous_chunk_groups(processed_chunks: List[Dict], context_words: List[str]) -> List[Dict[str, Any]]:
    """
    Tìm các nhóm chunk liên tiếp chứa context liên tục
    """
    results = []
    
    # Kiểm tra các nhóm 2 chunk liên tiếp
    for i in range(len(processed_chunks) - 1):
        chunk1 = processed_chunks[i]
        chunk2 = processed_chunks[i + 1]
        
        # Kết hợp text của 2 chunk
        combined_words = chunk1['processed_words'] + chunk2['processed_words']
        combined_text = ' '.join(combined_words)
        
        # Kiểm tra subsequence
        max_match_length = 0
        max_match_start = 0
        
        for start in range(len(combined_words) - len(context_words) + 1):
            match_length = 0
            for j in range(len(context_words)):
                if start + j < len(combined_words) and combined_words[start + j] == context_words[j]:
                    match_length += 1
                else:
                    break
            
            if match_length > max_match_length:
                max_match_length = match_length
                max_match_start = start
        
        # Nếu match tốt (ít nhất 90%)
        if max_match_length >= len(context_words) * 0.9:
            match_percentage = (max_match_length / len(context_words)) * 100
            
            # Xác định chunk nào chứa phần nào của match
            chunk1_end = len(chunk1['processed_words'])
            if max_match_start < chunk1_end and max_match_start + max_match_length <= chunk1_end:
                # Toàn bộ match nằm trong chunk1
                chunks_involved = [chunk1['original']]
                chunk_indices = [i]
                match_type = 'single_chunk_actually'
            elif max_match_start < chunk1_end:
                # Match trải qua 2 chunk
                chunks_involved = [chunk1['original'], chunk2['original']]
                chunk_indices = [i, i + 1]
                match_type = 'two_chunks'
            else:
                # Match nằm trong chunk2
                chunks_involved = [chunk2['original']]
                chunk_indices = [i + 1]
                match_type = 'single_chunk_actually'
            
            results.append({
                'type': match_type,
                'chunks': chunks_involved,
                'chunk_indices': chunk_indices,
                'score_info': {
                    'match_type': 'continuous',
                    'matched_words': max_match_length,
                    'total_context_words': len(context_words),
                    'match_percentage': match_percentage,
                    'position': max_match_start,
                    'matched_sequence': ' '.join(combined_words[max_match_start:max_match_start + max_match_length])
                }
            })
    
    # Sắp xếp theo match_percentage giảm dần
    results.sort(key=lambda x: x['score_info']['match_percentage'], reverse=True)
    
    # Lọc kết quả trùng lặp
    unique_results = []
    seen_chunk_sets = set()
    
    for result in results:
        chunk_ids = tuple(sorted(chunk.get('chunk_id', idx) for idx, chunk in zip(result['chunk_indices'], result['chunks'])))
        
        if chunk_ids not in seen_chunk_sets:
            unique_results.append(result)
            seen_chunk_sets.add(chunk_ids)
    
    return unique_results

def find_chunks_with_context_preprocessed(file_path: str, search_context: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Tìm chunk chứa context với preprocessing nâng cao
    """
    try:
        # Bước 1: Đọc và preprocess chunks
        print("🔧 Đang preprocess chunks...")
        with open(file_path, 'r', encoding='utf-8') as f:
            original_chunks = json.load(f)
        
        # Tạo file preprocessed
        processed_file = "Chunked/fixed/all_chunks_preprocessed.json"
        processed_chunks = save_preprocessed_chunks(original_chunks, processed_file)
        
        # Bước 2: Preprocess context
        print("🔧 Đang preprocess context...")
        context_processed = preprocess_text_advanced(search_context)
        context_words = context_processed.split()
        
        print(f"📊 Context đã xử lý: {len(context_words)} từ")
        print(f"   Sample: {' '.join(context_words[:15])}...")
        
        # Bước 3: Tìm exact match trong từng chunk
        print("\n🔍 Tìm exact match trong từng chunk...")
        exact_matches = find_exact_context_match(processed_chunks, context_processed, context_words)
        
        if exact_matches:
            print(f"✅ Tìm thấy {len(exact_matches)} chunk có exact match")
            # Chỉ lấy kết quả tốt nhất nếu có nhiều
            exact_matches.sort(key=lambda x: x['score_info']['match_percentage'], reverse=True)
            return exact_matches[:top_k]
        
        # Bước 4: Tìm trong các nhóm chunk liên tiếp
        print("🔍 Tìm trong các nhóm chunk liên tiếp...")
        continuous_matches = find_continuous_chunk_groups(processed_chunks, context_words)
        
        if continuous_matches:
            print(f"✅ Tìm thấy {len(continuous_matches)} nhóm chunk có match liên tiếp")
            return continuous_matches[:top_k]
        
        # Bước 5: Nếu không tìm thấy, tìm chunk có nhiều từ khóa nhất
        print("🔍 Tìm chunk có nhiều từ khóa nhất...")
        keyword_chunks = []
        
        # Tạo từ điển từ khóa từ context (loại bỏ stop words đơn giản)
        stop_words = {'cua', 'duoc', 'hoac', 'hoặc', 'các', 'cua', 'cho', 'về', 'trong', 'của'}
        keywords = [word for word in context_words if word not in stop_words and len(word) > 2]
        
        print(f"📊 Từ khóa quan trọng ({len(keywords)} từ): {', '.join(keywords[:10])}...")
        
        for idx, chunk in enumerate(processed_chunks):
            chunk_word_set = set(chunk['processed_words'])
            keyword_matches = sum(1 for kw in keywords if kw in chunk_word_set)
            
            if keyword_matches > len(keywords) * 0.5:  # Ít nhất 50% từ khóa
                keyword_chunks.append({
                    'type': 'keyword_match',
                    'chunks': [chunk['original']],
                    'chunk_indices': [idx],
                    'score_info': {
                        'match_type': 'keyword',
                        'keyword_matches': keyword_matches,
                        'total_keywords': len(keywords),
                        'match_percentage': (keyword_matches / len(keywords)) * 100,
                        'keywords_found': [kw for kw in keywords if kw in chunk_word_set]
                    }
                })
        
        if keyword_chunks:
            keyword_chunks.sort(key=lambda x: x['score_info']['keyword_matches'], reverse=True)
            print(f"✅ Tìm thấy {len(keyword_chunks)} chunk có từ khóa")
            return keyword_chunks[:top_k]
        
        print("❌ Không tìm thấy chunk nào phù hợp")
        return []
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return []

def print_enhanced_results(matches: List[Dict[str, Any]], original_context: str):
    """In kết quả chi tiết"""
    if not matches:
        print("Không tìm thấy chunk nào khớp với context")
        return
    
    print(f"\n{'='*100}")
    print(f"🏆 KẾT QUẢ TÌM KIẾM CHO CONTEXT")
    print(f"{'='*100}")
    print(f"📝 Context gốc ({len(original_context)} ký tự):")
    print(f"   {original_context[:150]}...\n")
    
    for idx, match in enumerate(matches, 1):
        score_info = match['score_info']
        
        print(f"{'─'*100}")
        print(f"🎯 Kết quả {idx} | Loại: {match['type']} | Match type: {score_info['match_type']}")
        print(f"{'─'*100}")
        
        print(f"📊 Độ khớp: {score_info.get('match_percentage', 0):.1f}%")
        
        if score_info['match_type'] in ['exact', 'subsequence', 'continuous']:
            print(f"   • Số từ khớp: {score_info['matched_words']}/{score_info['total_context_words']}")
            print(f"   • Vị trí bắt đầu: {score_info.get('position', 'N/A')}")
            
            if 'matched_sequence' in score_info:
                matched_seq = score_info['matched_sequence']
                print(f"\n   📝 Phần khớp được:")
                if len(matched_seq) > 200:
                    print(f"      \"{matched_seq[:200]}...\"")
                else:
                    print(f"      \"{matched_seq}\"")
        
        elif score_info['match_type'] == 'keyword':
            print(f"   • Từ khóa khớp: {score_info['keyword_matches']}/{score_info['total_keywords']}")
            if 'keywords_found' in score_info:
                print(f"   • Các từ khóa tìm thấy: {', '.join(score_info['keywords_found'][:10])}")
                if len(score_info['keywords_found']) > 10:
                    print(f"     ...và {len(score_info['keywords_found']) - 10} từ khác")
        
        print(f"\n📁 Chunks liên quan ({len(match['chunks'])} chunk):")
        
        for i, chunk in enumerate(match['chunks'], 1):
            chunk_idx = match['chunk_indices'][i-1] if i-1 < len(match['chunk_indices']) else 'N/A'
            print(f"\n   {'▸' if len(match['chunks']) > 1 else '━'} Chunk {i} (Index: {chunk_idx})")
            print(f"      ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"      File: {chunk.get('source_file', 'N/A')}")
            
            # Hiển thị text
            text = chunk.get('text', '')
            
            # Highlight phần khớp nếu có
            if 'matched_sequence' in score_info and score_info['matched_sequence']:
                # Tìm vị trí của matched_sequence trong text (đơn giản hóa)
                matched_lower = score_info['matched_sequence'].lower()
                text_lower = preprocess_text_advanced(text)
                
                if matched_lower in text_lower:
                    pos = text_lower.find(matched_lower)
                    if pos >= 0:
                        # Hiển thị với context xung quanh
                        start = max(0, pos - 50)
                        end = min(len(text), pos + len(matched_lower) + 50)
                        
                        preview = text[start:end]
                        if start > 0:
                            preview = "..." + preview
                        if end < len(text):
                            preview = preview + "..."
                        
                        print(f"      Text: {preview}")
                    else:
                        if len(text) > 300:
                            print(f"      Text: {text[:300]}...")
                        else:
                            print(f"      Text: {text}")
                else:
                    if len(text) > 300:
                        print(f"      Text: {text[:300]}...")
                    else:
                        print(f"      Text: {text}")
            else:
                if len(text) > 300:
                    print(f"      Text: {text[:300]}...")
                else:
                    print(f"      Text: {text}")
        
        print()

# Sử dụng
if __name__ == "__main__":
    if file_find and context:
        print(f"🔍 Đang tìm kiếm trong: {file_find}")
        print(f"📝 Context ({len(context.split())} từ, {len(context)} ký tự)\n")
        
        results = find_chunks_with_context_preprocessed(file_find, context, top_k=5)
        print_enhanced_results(results, context)
        
        # Thống kê
        if results:
            print(f"{'='*100}")
            print("📈 PHÂN TÍCH KẾT QUẢ:")
            
            match_types = {}
            for result in results:
                mt = result['score_info']['match_type']
                match_types[mt] = match_types.get(mt, 0) + 1
            
            for mt, count in match_types.items():
                print(f"   • {mt}: {count} kết quả")
            
            avg_match = sum(r['score_info'].get('match_percentage', 0) for r in results) / len(results)
            print(f"   • Độ khớp trung bình: {avg_match:.1f}%")
            
            best_result = results[0]
            best_match = best_result['score_info'].get('match_percentage', 0)
            print(f"   • Kết quả tốt nhất: {best_match:.1f}% khớp ({best_result['type']})")
            
            print(f"\n💡 Gợi ý:")
            if avg_match < 80:
                print("   - Context có thể không có trong dữ liệu chunk hiện tại")
                print("   - Thử tìm kiếm với các từ khóa chính")
            elif best_match >= 95:
                print("   - Tìm thấy chunk khớp rất tốt với context")
            else:
                print("   - Tìm thấy chunk có độ khớp khá")
                
    else:
        print("⚠️  Vui lòng cung cấp file_find và context để tìm kiếm")