import subprocess
import sys

packages = [
    'torch',
    'torchvision',
    'openai-clip',
    'faiss-cpu',
    'peft',
    'pillow',
    'scikit-learn',
    'tqdm',
    'numpy',
    'matplotlib',
    'deep-translator',
    'sentence_transformers',
    'clip',
    'streamlit',
    'pillow'
]

for package in packages:
    try:
        __import__(package.replace('-', '_'))
        print(f'{package} already installed')
    except ImportError:
        print(f'Installing {package}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
        print(f'{package} installed')