"""
RAG (Retrieval Augmented Generation) module for medical board items.
Provides semantic search over patient data, lab results, medications, and clinical notes.
"""

import json
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import canvas_ops
load_dotenv()

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID")
PROJECT_LOCATION = os.getenv("PROJECT_LOCATION", "us-central1")
EMBEDDING_MODEL = "text-embedding-005"
EMBEDDING_DIM = 768
INDEX_CACHE_PATH = "output/rag_index.pkl"

# Initialize client
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=PROJECT_LOCATION,
)


def load_board_items(file_path: str = "output/board_items.json") -> List[Dict[str, Any]]:
    """Load board items from JSON file."""

    try:
        return canvas_ops.get_board_items()
    except Exception as e:
        print(f"Error loading board items: {e}")
        with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    


def compute_item_hash(item: Dict[str, Any]) -> str:
    """Compute hash of item content to detect changes."""
    # Create a stable string representation of the item
    item_str = json.dumps(item, sort_keys=True)
    return hashlib.md5(item_str.encode()).hexdigest()


def save_index(index_df: pd.DataFrame, file_path: str = INDEX_CACHE_PATH) -> None:
    """Save index to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(index_df, f)
    print(f"Index saved to {file_path}")


def load_index(file_path: str = INDEX_CACHE_PATH) -> Optional[pd.DataFrame]:
    """Load index from disk if it exists."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                index_df = pickle.load(f)
            print(f"Loaded existing index with {len(index_df)} items")
            return index_df
        except Exception as e:
            print(f"Error loading index: {e}")
            return None
    return None


def extract_text_recursive(obj: Any, parent_key: str = "") -> List[str]:
    """
    Recursively extract text from nested dictionaries and lists.
    Returns a list of strings in format 'key: value'.
    """
    texts = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Skip certain keys that don't add value
            if key in ['x', 'y', 'width', 'height', 'color', 'rotation', 'createdAt', 'updatedAt']:
                continue
            
            # Create a readable key path
            full_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, (dict, list)):
                # Recurse into nested structures
                texts.extend(extract_text_recursive(value, full_key))
            elif value is not None and str(value).strip():
                # Add leaf values
                texts.append(f"{full_key}: {value}")
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                texts.extend(extract_text_recursive(item, parent_key))
            elif item is not None and str(item).strip():
                texts.append(f"{parent_key}: {item}")
    
    return texts


def extract_searchable_text(item: Dict[str, Any]) -> str:
    """
    Extract searchable text from a board item by recursively extracting all key-value pairs.
    """
    # Extract all text recursively
    text_parts = extract_text_recursive(item)
    
    # Join with separator
    return " | ".join(text_parts)


def get_embeddings(text: str, output_dim: int = EMBEDDING_DIM) -> Optional[List[float]]:
    """Generate embeddings for text using Vertex AI."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[text],
            config=EmbedContentConfig(output_dimensionality=output_dim),
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def build_index(board_items: List[Dict[str, Any]], existing_index: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Build searchable index from board items with incremental updates.
    
    Args:
        board_items: List of board items to index
        existing_index: Existing index DataFrame (if available)
    
    Returns:
        Tuple of (updated_index_df, stats_dict)
    """
    # Create a map of existing items by ID
    existing_items = {}
    if existing_index is not None and len(existing_index) > 0:
        for idx, row in existing_index.iterrows():
            item_id = row["id"]
            item_hash = compute_item_hash(row["item"])
            existing_items[item_id] = {
                "hash": item_hash,
                "row": row
            }
    
    stats = {
        "total": len(board_items),
        "new": 0,
        "updated": 0,
        "unchanged": 0
    }
    
    index_data = []
    
    for item in board_items:
        item_id = item.get("id", "")
        item_hash = compute_item_hash(item)
        searchable_text = extract_searchable_text(item)
        
        if not searchable_text.strip():
            print("skipping empty", item_id)
            continue
        
        # Check if item exists and hasn't changed
        if item_id in existing_items:
            if existing_items[item_id]["hash"] == item_hash:
                # Item unchanged, reuse existing data
                stats["unchanged"] += 1
                index_data.append(existing_items[item_id]["row"].to_dict())
                continue
            else:
                # Item updated
                stats["updated"] += 1
                print(f"Updating: {item_id[:50]}...")
        else:
            # New item
            stats["new"] += 1
            print(f"Adding new: {item_id[:50]}...")
        
        # Generate embeddings for new or updated items
        embeddings = get_embeddings(searchable_text)
        
        if embeddings:
            index_data.append({
                "id": item_id,
                "text": searchable_text,
                "embeddings": embeddings,
                "item": item,
                "hash": item_hash
            })
    
    return pd.DataFrame(index_data), stats


def search(query: str, index_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the index for relevant items.
    Returns top_k most relevant items with similarity scores.
    """
    # Get query embedding
    query_embedding = get_embeddings(query)
    if not query_embedding:
        return []
    
    # Calculate cosine similarity
    query_emb_array = np.array([query_embedding])
    index_embeddings = np.array(index_df["embeddings"].tolist())
    
    similarities = cosine_similarity(query_emb_array, index_embeddings)[0]
    
    # Get top k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "id": index_df.iloc[idx]["id"],
            "text": index_df.iloc[idx]["text"],
            "similarity": float(similarities[idx]),
            "item": index_df.iloc[idx]["item"]
        })
    
    return results


def format_context_for_llm(search_results: List[Dict[str, Any]]) -> str:
    """
    Format search results into context string for LLM.
    """
    context_parts = []
    
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"[Result {i}] (Relevance: {result['similarity']:.2f})")
        context_parts.append(result['text'])
        context_parts.append("")  # Empty line for separation
    
    return "\n".join(context_parts)


# Main RAG functions
def initialize_rag(data_path: str = "output/board_items.json", force_rebuild: bool = False) -> pd.DataFrame:
    """
    Initialize RAG system by loading data and building/updating index.
    
    Args:
        data_path: Path to board items JSON file
        force_rebuild: If True, rebuild index from scratch
    
    Returns:
        The index DataFrame
    """
    print("Loading board items...")
    board_items = load_board_items(data_path)
    print(f"Loaded {len(board_items)} items")
    
    # Try to load existing index
    existing_index = None if force_rebuild else load_index()
    
    if existing_index is not None:
        print("Checking for updates...")
    else:
        print("Building search index from scratch...")
    
    index_df, stats = build_index(board_items, existing_index)
    
    print(f"\nIndex statistics:")
    print(f"  Total items: {stats['total']}")
    print(f"  New items: {stats['new']}")
    print(f"  Updated items: {stats['updated']}")
    print(f"  Unchanged items: {stats['unchanged']}")
    print(f"  Final index size: {len(index_df)} searchable items")
    
    # Save the updated index
    save_index(index_df)
    
    return index_df


def query_rag(query: str, index_df: pd.DataFrame, top_k: int = 3, return_raw: bool = False):
    """
    Query the RAG system and return formatted context.
    
    Args:
        query: User's question
        index_df: Pre-built index DataFrame
        top_k: Number of results to return
        return_raw: If True, return raw results; if False, return formatted context
    
    Returns:
        Formatted context string for LLM or raw results list
    """
    print(f"Searching for: {query}")
    results = search(query, index_df, top_k=top_k)
    
    if not results:
        return [] if return_raw else "No relevant information found."
    
    if return_raw:
        return results
    
    # context = format_context_for_llm(results)
    return results


def run_rag(query,top_k=2):
    index = initialize_rag()
    results = query_rag(query, index, top_k=top_k, return_raw=True)
    return results

# Example usage
if __name__ == "__main__":
    import sys
    
    # Check for force rebuild flag
    force_rebuild = "--rebuild" in sys.argv
    
    # Initialize RAG
    index = initialize_rag(force_rebuild=force_rebuild)
    
    # Example queries
    test_queries = [
        "Give me summary of the patient",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = query_rag(query, index, top_k=2, return_raw=True)
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}] (Relevance: {result['similarity']:.3f})")
            print(f"ID: {result['id']}")
            with open("output/rag_results.json", "w") as f:
                json.dump(results, f, indent=4)