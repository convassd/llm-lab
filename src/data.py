from typing import Dict, List
from pathlib import Path

def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return 'We train a tiny bigram model.'
    return p.read_text(encoding='utf-8')

def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos

def encode(text: str, stoi: Dict[str, int]):
    return [stoi[c] for c in text]

def decode(ids: List[int], itos):
    return ''.join(itos[i] for i in ids)

def make_bigram_dataset(ids):
    xs, ys = [], []
    for i in range(len(ids) - 1):
        xs.append(ids[i])
        ys.append(ids[i + 1])
    import numpy as np
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)
