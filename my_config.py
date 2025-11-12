import torch
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = r'/workspaces/clip-coop-cocoop-custom/'

IMAGE_PATH = os.path.join(ROOT_DIR, 'new_image_paths.npy')
EMB_PATH = os.path.join(ROOT_DIR, 'new_embeddings.npy')

INDEX_FILE = os.path.join(ROOT_DIR, 'vector.index')
INDEX_FILE_DAI = os.path.join(ROOT_DIR, 'vector_finetuned.index')
OUTPUT_INDEX_PATH_FINE = os.path.join(ROOT_DIR, 'vector_finetuned.index')
PROMPTS_FILE = os.path.join(ROOT_DIR, 'fungi_prompts.json')
LORA_MODEL_DIR = os.path.join(ROOT_DIR, 'clip_text_lora_adapter')
LORA_MODEL_DIR_DAI = os.path.join(ROOT_DIR, 'clip_text_lora_adapter_dai')
TRINH_PATH = "Dataset_trinh"
TOP_K = 5

PHONG_MODEL = "CLIP + LoRa + Prompt tuning"
DAI_MODEL = "CLIP + LoRA"
TRINH_MODEL = "CLIP + reranking"

CLIP_PROMPT_TEMPLATES = [
    "a photo of {}",
    "a close-up photo of {}",
    "a natural photo of {}",
    "a picture of {}",
    "an image of {} in the wild"
]