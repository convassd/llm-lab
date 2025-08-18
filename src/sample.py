# src/sample.py
from pathlib import Path
import json
import yaml
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import BigramLM
from src.data import load_text, build_vocab, decode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "experiments/week02/config.yaml"

def sample_next_token_from_logits(
    logits_last_step: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # ensure shape (V,)
    if logits_last_step.dim() == 2:
        logits_last_step = logits_last_step.squeeze(0)

    if temperature is None:
        temperature = 1.0

    if temperature <= 0:
        return torch.argmax(logits_last_step, dim=-1, keepdim=False)

    logits = logits_last_step / float(temperature)
    V = logits.size(-1)

    # top-k
    if top_k is not None and 0 < top_k < V:
        top_vals, top_idx = torch.topk(logits, top_k)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(0, top_idx, top_vals)
        logits = masked

    # top-p (nucleus)
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep_sorted = cum <= top_p
        keep_sorted[..., 0] = True  # always keep top-1
        keep_idx = sorted_idx[keep_sorted]
        masked = torch.full_like(logits, float("-inf"))
        masked[keep_idx] = logits[keep_idx]
        logits = masked

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_id.squeeze(0)

def load_model_and_vocab(ckpt_path: Path, device: torch.device):
    # load state dict (weights_only=True is safer with your own checkpoints)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    # infer vocab size from weights (for bigram LM it's square [V,V])
    W = state["embed.weight"]
    vocab_size = int(W.shape[0])

    model = BigramLM(vocab_size=vocab_size).to(device)
    model.load_state_dict(state)
    model.eval()

    # load vocab.json saved by training
    vocab_json = ckpt_path.parent / "vocab.json"
    if vocab_json.exists():
        with open(vocab_json, "r", encoding="utf-8") as f:
            chars = json.load(f)["chars"]
    else:
        # fallback: rebuild from corpus (less safe)
        raise FileNotFoundError(f"Missing vocab.json at {vocab_json}")

    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in enumerate(chars)}
    return model, stoi, itos

def generate_text(
    ckpt_path: Path,
    *,
    start_ids: list[int] | None,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
):
    model, stoi, itos = load_model_and_vocab(ckpt_path, device)
    V = len(itos)

    if start_ids and len(start_ids) > 0:
        start_ids = [int(i) % V for i in start_ids]
        idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    else:
        # if nothing provided, fall back to first char in corpus (from vocab order)
        idx = torch.tensor([[0]], dtype=torch.long, device=device)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            logits = model(idx)  # (B,T,V) or (B,V) depending on your impl
            logits_last = logits[:, -1, :] if logits.dim() == 3 else logits
            next_id = sample_next_token_from_logits(
                logits_last.squeeze(0),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=gen,
            ).view(1, 1)
            idx = torch.cat([idx, next_id], dim=1)

    tokens = idx[0].tolist()
    text = decode(tokens, itos)
    return text

def main():
    # --- load config ---
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- resolve device ---
    dev = cfg.get("device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    # derive checkpoint path from corpus_path
    corpus_path = PROJECT_ROOT / cfg["corpus_path"]
    ckpt_path = corpus_path.parent / "checkpoints" / "bigram.pt"
    save_dir = PROJECT_ROOT / cfg["save_dir"]
    sample_cfg = cfg.get("sample", {}) or {}
    start_char = sample_cfg.get("start_char", "")
    max_new_tokens = int(sample_cfg.get("max_new_tokens", 300))
    temperature = float(sample_cfg.get("temperature", 1.0))
    top_k = sample_cfg.get("top_k", None)
    top_p = sample_cfg.get("top_p", None)
    seed = int(cfg.get("seed", 1337))

    # build start_ids from start_char using saved vocab
    # (we load stoi via load_model_and_vocab below, but we need it now; so read vocab.json directly)
    vocab_json = ckpt_path.parent / "vocab.json"
    with open(vocab_json, "r", encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}

    start_ids = None
    if start_char:
        # filter to known chars (for bigram only last char matters, but accept full string)
        filtered = [c for c in start_char if c in stoi]
        if filtered:
            start_ids = [stoi[c] for c in filtered]

    sample_text = generate_text(
        ckpt_path=ckpt_path,
        start_ids=start_ids,
        max_new_tokens=max_new_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )
    (save_dir / "sample.txt").write_text(sample_text, encoding="utf-8")
    print(f"Saved sample text to {save_dir / 'sample.txt'}")

if __name__ == "__main__":
    main()
