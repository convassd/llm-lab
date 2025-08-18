# src/train.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import json
import yaml
import torch
import torch.nn as nn
from tqdm import trange

from src.data import load_text, build_vocab, encode, decode, make_bigram_dataset
from src.model import BigramLM
from src.utils import set_seed, save_plot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "experiments/week02/config.yaml"

def main():
    # --- load config ---
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- resolve device ---
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # --- seed for reproducibility ---
    set_seed(int(cfg.get("seed", 1337)))

    # --- data ---
    corpus_path = PROJECT_ROOT / cfg["corpus_path"]
    text = load_text(str(corpus_path))
    print(f"Training with corpus file: {corpus_path}")
    chars, stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    X, Y = make_bigram_dataset(ids)

    x = torch.tensor(X, dtype=torch.long, device=device)
    y = torch.tensor(Y, dtype=torch.long, device=device)

    # --- model/optim ---
    model = BigramLM(vocab_size=len(chars)).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.CrossEntropyLoss()

    # --- train ---
    steps = int(cfg["steps"])
    batch_size = int(cfg["batch_size"])
    log_steps, log_losses = [], []
    model.train()

    for step in trange(steps, desc="training"):
        idx = torch.randint(0, x.shape[0], (batch_size,), device=device)
        xb, yb = x[idx], y[idx]
        logits = model(xb)                 # (B, V) for bigram, or (B,T,V) depending on your impl
        loss = loss_fn(logits, yb)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (step + 1) % 50 == 0:
            log_steps.append(step + 1)
            log_losses.append(float(loss.item()))

    # --- save artifacts (under experiments/<week02>/...) ---
    save_dir = corpus_path.parent                     # e.g., experiments/week01
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # 1) loss plot
    save_plot(str(save_dir / "loss.png"), log_steps, log_losses)

    # 2) checkpoint (state_dict)
    ckpt_path = save_dir / "checkpoints" / "bigram.pt"
    torch.save(model.state_dict(), ckpt_path)

    # 3) vocab.json
    with open(save_dir / "checkpoints" / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"chars": chars}, f, ensure_ascii=False, indent=2)

    # 4) quick sample (greedy for determinism)
    model.eval()
    with torch.no_grad():
        start_id = encode(text[:1], stoi)[0]   # first character of corpus
        # simple greedy generation using model directly
        idx_gen = torch.tensor([[start_id]], dtype=torch.long, device=device)
        for _ in range(300):
            logits = model(idx_gen)
            logits_last = logits[:, -1, :] if logits.dim() == 3 else logits  # handle (B,T,V) vs (B,V)
            next_id = torch.argmax(logits_last, dim=-1, keepdim=True)        # greedy
            if logits.dim() == 3:
                idx_gen = torch.cat([idx_gen, next_id], dim=1)
            else:
                # bigram that returns (B,V): keep a 2D sequence
                idx_gen = torch.cat([idx_gen, next_id.view(1, 1)], dim=1)

    train_text = decode(idx_gen[0].tolist(), itos)
    print(train_text)

    print(f"Saved loss plot to {save_dir / 'loss.png'}")
    print(f"Saved weights to {ckpt_path}")

if __name__ == "__main__":
    main()
