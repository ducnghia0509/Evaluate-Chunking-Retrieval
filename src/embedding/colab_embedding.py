# Google Colab Embedding Script
# Upload folder Chunked và chạy script này để tạo embeddings
# Output: folder Embedd với cùng cấu trúc

# ============== SETUP ==============
# Run this cell first
!pip install sentence-transformers transformers torch

# ============== IMPORTS ==============
import json
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np

# ============== MOUNT GOOGLE DRIVE (Optional) ==============
# Uncomment if you want to use Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# ============== CONFIGURATION ==============
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
CHUNKED_DIR = "/content/Chunked"  # Change this to your uploaded folder path
EMBEDD_DIR = "/content/Embedd"    # Output folder

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============== LOAD MODEL ==============
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(
    MODEL_NAME,
    device=device,
    trust_remote_code=True
)
print("Model loaded successfully!")

# ============== EMBEDDING FUNCTION ==============
def embed_chunks(chunks_data, batch_size=32):
    """
    Embed all chunks in a JSON file
    
    Args:
        chunks_data: List of chunk dictionaries
        batch_size: Batch size for encoding
    
    Returns:
        chunks_data with embeddings added
    """
    # Extract texts
    texts = [chunk['text'] for chunk in chunks_data]
    
    # Generate embeddings in batches
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks_data, embeddings):
        chunk['embedding'] = embedding.tolist()
        chunk['embedding_model'] = MODEL_NAME
        chunk['embedding_dim'] = len(embedding)
    
    return chunks_data


# ============== PROCESS ALL FILES ==============
def process_all_chunked_files():
    """Process all JSON files in Chunked folder and save to Embedd folder"""
    
    chunked_path = Path(CHUNKED_DIR)
    embedd_path = Path(EMBEDD_DIR)
    
    if not chunked_path.exists():
        print(f"Error: Chunked directory not found: {CHUNKED_DIR}")
        print("Please upload your Chunked folder to Colab")
        return
    
    # Create output directory
    embedd_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(chunked_path.rglob('*.json'))
    print(f"\nFound {len(json_files)} JSON files to process\n")
    
    stats = {
        'total_files': len(json_files),
        'total_chunks': 0,
        'processed_files': 0
    }
    
    # Process each file
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            # Read chunks
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Skip if not a list or empty
            if not isinstance(chunks_data, list) or len(chunks_data) == 0:
                print(f"Skipping {json_file.name}: invalid format or empty")
                continue
            
            print(f"\nProcessing: {json_file.name}")
            
            # Embed chunks
            chunks_data = embed_chunks(chunks_data)
            
            # Create output path with same structure
            relative_path = json_file.relative_to(chunked_path)
            output_file = embedd_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with embeddings
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            stats['total_chunks'] += len(chunks_data)
            stats['processed_files'] += 1
            
            print(f"  ✓ Saved: {output_file}")
            
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING COMPLETE!")
    print("="*60)
    print(f"Total files processed: {stats['processed_files']}/{stats['total_files']}")
    print(f"Total chunks embedded: {stats['total_chunks']}")
    print(f"Output directory: {EMBEDD_DIR}")
    print("="*60)
    
    return stats


# ============== RUN ==============
if __name__ == "__main__":
    stats = process_all_chunked_files()
    
    # Optionally, download the Embedd folder
    # from google.colab import files
    # !zip -r Embedd.zip {EMBEDD_DIR}
    # files.download('Embedd.zip')
