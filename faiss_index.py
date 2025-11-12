import argparse
import clip
import torch
import torch.nn.functional as F
from PIL import Image
from glob import glob
import os
from pathlib import Path
from tqdm import tqdm
import faiss
import numpy as np
import json
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
import shutil
import matplotlib.pyplot as plt
import math
import textwrap
from math import log2
from collections import Counter
import zipfile
from os.path import basename
def create_faiss_index(embeddings, image_paths, output_path):
    embeddings = np.asarray(embeddings).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embeddings = embeddings / norms

    d = embeddings.shape[1]

    index = faiss.IndexFlatL2(d)

    index = faiss.IndexIDMap(index)
    ids = np.arange(len(embeddings)).astype(np.int64)
    index.add_with_ids(embeddings, ids)

    output_path = Path(output_path)
    faiss.write_index(index, str(output_path))
    print(f'FAISS FlatL2 index saved to {output_path}')

    with open(str(output_path) + '.paths', 'w', encoding='utf-8') as f:
      for p in image_paths:
        f.write(str(p) + '\n')

    return index

def load_faiss_index(index_path):
  index = faiss.read_index(str(index_path))
  with open(str(index_path) + '.paths', 'r', encoding='utf-8') as f:
    image_paths = [line.strip() for line in f]

  print(f'FAISS FlatL2 index loaded from {index_path}')
  return index, image_paths