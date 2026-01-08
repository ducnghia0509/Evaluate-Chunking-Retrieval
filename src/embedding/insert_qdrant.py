"""
Insert embedded chunks from Embedd folder into Qdrant
Creates separate collections for each chunking strategy
"""

import json
from pathlib import Path
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm


# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDD_DIR = "../../Embedd"

# Single collection for all strategies
COLLECTION_NAME = "evaluate"

# Default vector size (adjust based on your embedding model)
VECTOR_SIZE = 768


def delete_all_collections(client: QdrantClient):
    """
    Delete all existing collections in Qdrant
    
    Args:
        client: Qdrant client
    """
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if not collection_names:
            print("  No existing collections to delete")
            return
        
        print(f"  Found {len(collection_names)} existing collections:")
        for name in collection_names:
            print(f"    - {name}")
        
        for name in collection_names:
            try:
                client.delete_collection(name)
                print(f"  ✓ Deleted collection: {name}")
            except Exception as e:
                print(f"  ✗ Failed to delete {name}: {e}")
    except Exception as e:
        print(f"  Error listing collections: {e}")


def create_collection(client: QdrantClient, collection_name: str):
    """
    Create Qdrant collection
    
    Args:
        client: Qdrant client
        collection_name: Name of collection to create
    """
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
            on_disk=False  # Keep in memory for faster search
        )
    )
    print(f"  ✓ Created collection: {collection_name}")


def insert_chunks_to_collection(client: QdrantClient, collection_name: str, chunks_data: List[Dict], 
                               strategy: str, category: str, file_name: str):
    """
    Insert chunks into specific collection
    
    Args:
        client: Qdrant client
        collection_name: Target collection name
        chunks_data: List of chunk dictionaries
        strategy: Chunking strategy name
        category: Document category
        file_name: Source file name
    
    Returns:
        Statistics for this insertion
    """
    stats = {
        'chunks_processed': 0,
        'chunks_inserted': 0,
        'chunks_skipped': 0
    }
    
    batch_points = []
    batch_size = 100
    
    # Get current collection count for ID generation
    try:
        collection_info = client.get_collection(collection_name)
        start_id = collection_info.points_count
    except:
        start_id = 0
    
    point_id = start_id
    
    for chunk in chunks_data:
        stats['chunks_processed'] += 1
        
        # Validate chunk has embedding
        if 'embedding' not in chunk:
            stats['chunks_skipped'] += 1
            continue
        
        # Prepare payload
        payload = {
            'chunk_id': chunk.get('chunk_id', ''),
            'text': chunk.get('text', ''),
            'source_file': chunk.get('source_file', ''),
            'chunk_index': chunk.get('chunk_index', 0),
            'chunk_type': chunk.get('chunk_type', 'standard'),
            
            # Strategy and category info
            'chunking_strategy': strategy,
            'category': category,
            'source_json': file_name,
            
            # Original metadata
            'metadata': chunk.get('metadata', {}),
            
            # Embedding info
            'embedding_model': chunk.get('embedding_model', 'unknown'),
            'embedding_dim': chunk.get('embedding_dim', VECTOR_SIZE),
        }
        
        # Add parent-child specific info
        if strategy == "parent_child":
            if 'metadata' in chunk and 'role' in chunk['metadata']:
                payload['chunk_role'] = chunk['metadata']['role']
                if chunk['metadata']['role'] == 'child':
                    payload['parent_id'] = chunk['metadata'].get('parent_id', '')
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=chunk['embedding'],
            payload=payload
        )
        
        batch_points.append(point)
        point_id += 1
        stats['chunks_inserted'] += 1
        
        # Batch insert
        if len(batch_points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            batch_points = []
    
    # Insert remaining points
    if batch_points:
        client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
    
    return stats


def process_embedd_folder(client: QdrantClient, embedd_dir: str):
    """
    Process all JSON files in Embedd folder and insert into appropriate collections
    
    Args:
        client: Qdrant client
        embedd_dir: Path to Embedd folder
    """
    embedd_path = Path(embedd_dir)
    
    if not embedd_path.exists():
        raise ValueError(f"Embedd directory not found: {embedd_dir}")
    
    # Find all JSON files
    json_files = list(embedd_path.rglob('*.json'))
    print(f"\nFound {len(json_files)} JSON files to insert")
    
    # Overall statistics
    overall_stats = {
        'total_files': len(json_files),
        'files_processed': 0,
        'files_skipped': 0,
        'total_chunks': 0,
        'total_inserted': 0,
        'total_skipped': 0,
        'strategies': {}
    }
    
    # Process each file
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            # Read chunks
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Skip if not a list or empty
            if not isinstance(chunks_data, list) or len(chunks_data) == 0:
                overall_stats['files_skipped'] += 1
                continue
            
            # Determine chunking strategy and category from path
            # Path structure: Embedd/<strategy>/<category>/<file>_chunks.json
            relative_path = json_file.relative_to(embedd_path)
            parts = relative_path.parts
            
            if len(parts) >= 1:
                strategy_folder = parts[0]  # fixed, structure_paragraph, etc.
            else:
                print(f"Skipping {json_file.name}: cannot determine strategy")
                overall_stats['files_skipped'] += 1
                continue
            
            # Determine category
            category = "unknown"
            if len(parts) >= 2:
                category = parts[1]
            
            # Insert chunks into single collection
            file_stats = insert_chunks_to_collection(
                client=client,
                collection_name=COLLECTION_NAME,
                chunks_data=chunks_data,
                strategy=strategy_folder,
                category=category,
                file_name=json_file.name
            )
            
            # Update overall statistics
            overall_stats['files_processed'] += 1
            overall_stats['total_chunks'] += file_stats['chunks_processed']
            overall_stats['total_inserted'] += file_stats['chunks_inserted']
            overall_stats['total_skipped'] += file_stats['chunks_skipped']
            
            # Track by strategy
            if strategy_folder not in overall_stats['strategies']:
                overall_stats['strategies'][strategy_folder] = {'chunks': 0, 'files': 0}
            overall_stats['strategies'][strategy_folder]['chunks'] += file_stats['chunks_inserted']
            overall_stats['strategies'][strategy_folder]['files'] += 1
            
        except Exception as e:
            print(f"\nError processing {json_file.name}: {e}")
            overall_stats['files_skipped'] += 1
    
    return overall_stats


def main():
    """Main function to insert embeddings into Qdrant with single collection"""
    print("="*70)
    print("QDRANT INSERTION SCRIPT - SINGLE COLLECTION")
    print("="*70)
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Embedd directory: {EMBEDD_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    print("="*70)
    
    # Initialize Qdrant client
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )
    
    # Test connection
    try:
        collections = client.get_collections()
        print(f"\nConnected to Qdrant successfully!")
        existing_collections = [c.name for c in collections.collections]
        if existing_collections:
            print(f"Existing collections: {', '.join(existing_collections)}")
        else:
            print("No existing collections")
    except Exception as e:
        print(f"\nError connecting to Qdrant: {e}")
        print("Make sure Qdrant is running (docker)")
        return
    
    # Delete all existing collections
    print("\n" + "-"*70)
    print("Deleting all existing collections...")
    delete_all_collections(client)
    
    # Create single collection
    print("\n" + "-"*70)
    print("Creating collection...")
    create_collection(client, COLLECTION_NAME)
    
    # Process and insert chunks
    print("\n" + "-"*70)
    stats = process_embedd_folder(client, EMBEDD_DIR)
    
    # Print summary
    print("\n" + "="*70)
    print("INSERTION COMPLETE!")
    print("="*70)
    print(f"Files processed: {stats['files_processed']}/{stats['total_files']}")
    print(f"Files skipped: {stats['files_skipped']}")
    print(f"\nChunks processed: {stats['total_chunks']}")
    print(f"Chunks inserted: {stats['total_inserted']}")
    print(f"Chunks skipped (no embedding): {stats['total_skipped']}")
    
    print("\nStrategy statistics:")
    print("-" * 40)
    for strategy_name, strategy_stats in stats['strategies'].items():
        if strategy_stats['chunks'] > 0:
            print(f"{strategy_name}:")
            print(f"  Files: {strategy_stats['files']}")
            print(f"  Chunks: {strategy_stats['chunks']}")
    
    # Verify collection
    print("\n" + "-"*70)
    print("Collection verification:")
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"  {COLLECTION_NAME}: {collection_info.points_count} points")
    
    print("\n✓ Done! All chunks inserted into single collection.")
    print(f"  Use 'chunking_strategy' field to filter by chunking method")


if __name__ == "__main__":
    main()