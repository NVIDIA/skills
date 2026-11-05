"""Small decoder-only LM training loop on synthetic tokens (single GPU)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

VOCAB, SEQ_LEN, BATCH_SIZE, MAX_STEPS = 32000, 1024, 32, 300


class TokenDataset(Dataset):
    def __len__(self):
        return BATCH_SIZE * MAX_STEPS

    def __getitem__(self, idx):
        tokens = torch.randint(0, VOCAB, (SEQ_LEN + 1,))
        return tokens[:-1], tokens[1:]


class TinyLM(nn.Module):
    def __init__(self, d_model=1024, n_layers=8, n_heads=16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, tokens):
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=tokens.device)
        return self.head(self.blocks(self.embed(tokens), mask=mask, is_causal=True))


def main():
    model = TinyLM().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    loader = DataLoader(TokenDataset(), batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    for step, (tokens, targets) in enumerate(loader):
        tokens = tokens.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step + 1 == MAX_STEPS:
            break


if __name__ == "__main__":
    main()
