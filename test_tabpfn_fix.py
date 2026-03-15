import torch
import os
import shutil
from tabpfn import TabPFNClassifier

# Patch torch.load for compatibility with PyTorch 2.6 security settings
original_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_torch_load

# Clear broken cache
cache_dir = os.path.expanduser('~/.cache/tabpfn')
if os.path.exists(cache_dir):
    print(f"Clearing cache: {cache_dir}")
    shutil.rmtree(cache_dir)

print("Attempting to initialize TabPFN and download model...")
try:
    classifier = TabPFNClassifier(device='cpu')
    print("SUCCESS: TabPFN initialized and model loaded.")
except Exception as e:
    print(f"FAILURE: {e}")
