from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.model import BigramLM
from src.data import decode, build_vocab, load_text

def sample_next_token(
    logits_last_step: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    logits_last_step: (V,) or (1, V) tensor of logits for the next token.
    returns: (1,) int64 token id
    """
    # ensure shape (V,)
    if logits_last_step.dim() == 2:
        logits_last_step = logits_last_step.squeeze(0)

    # temperature
    if temperature <= 0:
        # "greedy" fallback if someone passes 0 or a negative
        return torch.argmax(logits_last_step, dim=-1, keepdim=False)

    logits = logits_last_step / float(temperature)

    V = logits.size(-1)

    # top-k: keep k highest logits
    if top_k is not None and 0 < top_k < V:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, topk_idx, topk_vals)
        logits = mask  # masked logits (others -inf)

    # top-p (nucleus): keep smallest set with cumulative prob >= p
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask_sorted = cumprobs <= top_p
        # always keep the largest-prob token
        keep_mask_sorted[..., 0] = True
        keep_idx = sorted_idx[keep_mask_sorted]
        # mask out everything not in nucleus
        mask = torch.full_like(logits, float("-inf"))
        mask[keep_idx] = logits[keep_idx]
        logits = mask

    # sample
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.argmax(probs, dim=-1)
    return next_id.squeeze(0)

def load_model_from_ckpt(ckpt_path, device="cpu"):
    device = torch.device(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    w = state["embed.weight"]                    # tensor with shape [V, V] in your bigram LM
    vocab_size = w.shape[0]                      # infer V
    model = BigramLM(vocab_size=vocab_size).to(device)
    model.load_state_dict(state)                 # shapes match now
    model.eval()
    return model, vocab_size

def generate_text(
    ckpt_path,
    start_token=0,
    max_new_tokens=50,
    device="cpu",
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
):
    model, vocab_size = load_model_from_ckpt(ckpt_path, device)
    start_token = int(start_token) % vocab_size

    # optional deterministic sampling seed
    gen = None
    if seed is not None:
        gen = torch.Generator(device=model.embed.weight.device).manual_seed(seed)

    idx = torch.tensor([[start_token]], dtype=torch.long, device=model.embed.weight.device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx)          # (B, T, V)
            logits = logits[:, -1, :]    # (B, V) → last step
            # sample one token with your knobs
            next_id = sample_next_token(
                logits.squeeze(0),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=gen,
            ).view(1, 1)
            idx = torch.cat([idx, next_id], dim=1)

    return idx[0].tolist()

if __name__ == "__main__":
    corpus_path = "experiments/week01/tiny_corpus.txt"
    text_corpus = load_text(corpus_path)           # read the file contents
    chars, stoi, itos = build_vocab(text_corpus)   # build vocab from text
    tokens = generate_text("experiments/week01/checkpoints/bigram.pt",
                           start_token=0, device="cuda")
    text = decode(tokens, itos)
    print("Generated text:", text)
    
    
