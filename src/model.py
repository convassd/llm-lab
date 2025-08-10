import torch
import torch.nn as nn

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        logits = self.embed(x)
        return logits

    @torch.no_grad()
    def generate(self, start_id: int, max_new_tokens: int = 200):
        device = next(self.parameters()).device
        x = torch.tensor([start_id], dtype=torch.long, device=device)
        out_ids = [start_id]
        for _ in range(max_new_tokens):
            logits = self.forward(x[-1:])
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1)
            out_ids.append(int(next_id.item()))
            x = torch.cat([x, next_id], dim=0)
        return out_ids
