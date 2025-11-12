import torch
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = r'/workspaces/clip-coop-cocoop-custom/'

IMAGE_PATH = os.path.join(ROOT_DIR, 'original_image_path.npy')
EMB_PATH = os.path.join(ROOT_DIR, 'original_embeddings.npy')

INDEX_FILE = os.path.join(ROOT_DIR, 'vector.index')
INDEX_FILE_DAI = os.path.join(ROOT_DIR, 'vector_finetuned.index')
OUTPUT_INDEX_PATH_FINE = os.path.join(ROOT_DIR, 'vector_finetuned.index')
PROMPTS_FILE = os.path.join(ROOT_DIR, 'fungi_prompts.json')
LORA_MODEL_DIR = os.path.join(ROOT_DIR, 'clip_text_lora_adapter')
LORA_MODEL_DIR_DAI = os.path.join(ROOT_DIR, 'clip_text_lora_adapter_dai')
TOP_K = 5