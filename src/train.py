# .venv\Scripts\Activate.ps1

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from tqdm import trange

from src.data import load_text, build_vocab, encode, decode, make_bigram_dataset
from src.model import BigramLM
from src.utils import set_seed, save_plot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='experiments/week01/tiny_corpus.txt')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='experiments/week01')
    args = parser.parse_args()

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text = load_text(args.corpus)
    chars, stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    X, Y = make_bigram_dataset(ids)

    x = torch.tensor(X, dtype=torch.long, device=device)
    y = torch.tensor(Y, dtype=torch.long, device=device)

    model = BigramLM(vocab_size=len(chars)).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    steps, losses = [], []
    model.train()
    for step in trange(args.steps, desc='training'):
        idx = torch.randint(0, x.shape[0], (256,), device=device)
        xb, yb = x[idx], y[idx]
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (step + 1) % 50 == 0:
            steps.append(step + 1)
            losses.append(float(loss.item()))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_plot(str(save_dir / 'loss.png'), steps, losses)

    with torch.no_grad():
        start_id = x[0].item()
        out_ids = model.generate(start_id, max_new_tokens=400)
    sample = decode(out_ids, itos)
    (save_dir / 'sample.txt').write_text(sample, encoding='utf-8')

    torch.save(model.state_dict(), save_dir / 'checkpoints' / 'bigram.pt')

    print(f'Saved loss plot to {save_dir / "loss.png"}')
    print(f'Saved sample text to {save_dir / "sample.txt"}')
    print(f'Saved weights to {save_dir / "checkpoints" / "bigram.pt"}')

if __name__ == '__main__':
    main()
