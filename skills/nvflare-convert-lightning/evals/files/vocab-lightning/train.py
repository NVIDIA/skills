# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch
from model import LitTextCNN
from torch.utils.data import DataLoader, TensorDataset

TEXTS = [
    "federated learning protects local data",
    "secure aggregation keeps updates private",
    "local training uses site specific records",
    "global models learn from many clients",
]
LABELS = [1, 1, 0, 0]


def build_vocab(texts):
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, length=8):
    ids = [vocab.get(token, vocab["<unk>"]) for token in text.split()]
    ids = ids[:length] + [vocab["<pad>"]] * max(0, length - len(ids))
    return ids


def make_loader(vocab):
    tokens = torch.tensor([encode(text, vocab) for text in TEXTS], dtype=torch.long)
    labels = torch.tensor(LABELS, dtype=torch.long)
    return DataLoader(TensorDataset(tokens, labels), batch_size=2)


def main():
    vocab = build_vocab(TEXTS)
    model = LitTextCNN(vocab_size=len(vocab))
    train_loader = make_loader(vocab)
    val_loader = make_loader(vocab)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
