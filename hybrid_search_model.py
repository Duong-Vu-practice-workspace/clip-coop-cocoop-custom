import torch
import numpy as np
import pandas as pd
import os
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

class Config:
    ASSETS_DIR = 'assets'
    IMAGE_VECTORS_PATH = os.path.join(ASSETS_DIR, 'image_vectors.npy')
    KNOWLEDGE_VECTORS_PATH = os.path.join(ASSETS_DIR, 'knowledge_vectors.npy')
    IMAGE_PATHS_JSON = os.path.join(ASSETS_DIR, 'image_paths.json')
    KNOWLEDGE_DF_PATH = os.path.join(ASSETS_DIR, 'knowledge_df.json')
    
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    ST_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Search parameters
    DEFAULT_ALPHA = 0.6
    DEFAULT_TOP_K = 10

# --- PUBLIC API ---

def load_all_resources():
    """
    Loads all models and pre-computed assets into a single dictionary.
    This function should be called once and its output cached.
    """
    print("--- Loading All Search Resources ---")
    
    # 1. Load Models
    print(f"Loading models on device '{Config.DEVICE}'...")
    clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME).to(Config.DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)
    st_model = SentenceTransformer(Config.ST_MODEL_NAME, device=Config.DEVICE)
    
    # 2. Load Pre-computed Assets
    print(f"Loading assets from '{Config.ASSETS_DIR}' directory...")
    image_vectors = np.load(Config.IMAGE_VECTORS_PATH)
    knowledge_vectors = np.load(Config.KNOWLEDGE_VECTORS_PATH)
    
    with open(Config.IMAGE_PATHS_JSON, 'r') as f:
        image_paths = json.load(f)
        
    knowledge_df = pd.read_json(Config.KNOWLEDGE_DF_PATH)
    
    # 3. Create necessary mappings for fast lookups
    species_to_knowledge_idx = {name: i for i, name in enumerate(knowledge_df['mushroom_name'])}
    
    # To map from image index to species name, we need to reconstruct this from the original CSV logic
    # This is a bit inefficient, but required if we don't save this mapping during asset build
    temp_df = pd.read_csv('full_dataset.csv')
    image_species_list = temp_df['mushroom_name'].tolist()


    print("--- All resources loaded successfully ---")

    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "st_model": st_model,
        "image_vectors": image_vectors,
        "knowledge_vectors": knowledge_vectors,
        "image_paths": image_paths,
        "knowledge_df": knowledge_df,
        "species_to_knowledge_idx": species_to_knowledge_idx,
        "image_species_list": image_species_list
    }

def search_by_text(query_text, resources, alpha=Config.DEFAULT_ALPHA, top_k=Config.DEFAULT_TOP_K):
    """
    Performs a hybrid search based on a text query.
    Returns a list of image file paths.
    """
    # Unpack resources
    q_clip = _encode_text_clip(query_text, resources)
    q_knowledge = _encode_text_sbert(query_text, resources)
    
    visual_scores = (resources["image_vectors"] @ q_clip.T).flatten()
    knowledge_scores = (resources["knowledge_vectors"] @ q_knowledge.T).flatten()

    # Re-ranking logic
    final_scores = np.zeros(len(resources["image_paths"]))
    for i, species_name in enumerate(resources["image_species_list"]):
        knowledge_idx = resources["species_to_knowledge_idx"].get(species_name)
        if knowledge_idx is not None:
            final_scores[i] = alpha * visual_scores[i] + (1 - alpha) * knowledge_scores[knowledge_idx]
            
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    
    return [resources["image_paths"][i] for i in top_indices]

def search_by_image(image_file, resources, top_k=Config.DEFAULT_TOP_K):
    """
    Performs an image-to-image search.
    'image_file' can be a file path or a file-like object (e.g., from Streamlit).
    Returns a list of image file paths.
    """
    try:
        query_image = Image.open(image_file).convert("RGB")
        query_vector = _encode_image_clip(query_image, resources)
        
        similarities = (resources["image_vectors"] @ query_vector.T).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [resources["image_paths"][i] for i in top_indices]
    except Exception as e:
        print(f"Error during image search: {e}")
        return []

# --- PRIVATE HELPER FUNCTIONS ---

def _encode_text_clip(text, resources):
    """Encodes a text query using CLIP."""
    with torch.no_grad():
        inputs = resources["clip_processor"](text=text, return_tensors="pt").to(Config.DEVICE)
        features = resources["clip_model"].get_text_features(**inputs)
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()

def _encode_text_sbert(text, resources):
    """Encodes a text query using SentenceTransformer."""
    embedding = resources["st_model"].encode([text], convert_to_numpy=True)
    embedding /= np.linalg.norm(embedding)
    return embedding

def _encode_image_clip(image, resources):
    """Encodes an image using CLIP."""
    with torch.no_grad():
        inputs = resources["clip_processor"](images=image, return_tensors="pt").to(Config.DEVICE)
        features = resources["clip_model"].get_image_features(**inputs)
    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()
