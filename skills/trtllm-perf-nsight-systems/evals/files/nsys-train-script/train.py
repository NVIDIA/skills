"""Minimal ResNet-50 training script (synthetic data) used for profiling.

Run: python train.py --iters 200
"""
import argparse

import torch
import torch.nn as nn
import torchvision


def get_loader(batch_size, num_batches):
    # Synthetic stand-in for an ImageNet DataLoader.
    for _ in range(num_batches):
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 1000, (batch_size,))
        yield images, labels


def train_step(model, optimizer, criterion, images, labels):
    images = images.cuda()
    labels = labels.cuda()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(images)
        loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    model = torchvision.models.resnet50().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for step, (images, labels) in enumerate(get_loader(args.batch_size, args.iters)):
        loss = train_step(model, optimizer, criterion, images, labels)
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
