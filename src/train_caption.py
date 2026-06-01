import os
import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

from vocab import Vocab
from dataset import CaptionDataset, make_clip_caption_collate_fn
from model import CLIPCaptionGRU


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    for pixel_values, ids, _ in tqdm(loader, leave=False):
        pixel_values = pixel_values.to(device)
        ids = ids.to(device)

        input_ids = ids[:, :-1]
        target_ids = ids[:, 1:]

        with torch.set_grad_enabled(train):
            logits = model(pixel_values, input_ids)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        non_pad = target_ids.ne(model.pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--token-embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--unfreeze-clip", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vocab = Vocab.load(os.path.join(args.data_dir, "vocab.json"))

    processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    train_dataset = CaptionDataset(
        os.path.join(args.data_dir, "train.csv"),
        vocab,
        max_len=args.max_len,
    )

    val_dataset = CaptionDataset(
        os.path.join(args.data_dir, "val.csv"),
        vocab,
        max_len=args.max_len,
    )

    collate_fn = make_clip_caption_collate_fn(
        processor=processor,
        pad_id=vocab.pad_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = CLIPCaptionGRU(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        clip_model_name=args.clip_model_name,
        token_embed_dim=args.token_embed_dim,
        hidden_dim=args.hidden_dim,
        freeze_clip=not args.unfreeze_clip,
    ).to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_ppl = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
        )

        val_loss, val_ppl = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            train=False,
        )

        print(
            f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f} | "
            f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "vocab_size": len(vocab),
            "pad_id": vocab.pad_id,
            "bos_id": vocab.bos_id,
            "eos_id": vocab.eos_id,
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.out_dir, "clip_caption_last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "clip_caption_best.pt"))
            print("Saved best checkpoint.")

    print("Training finished.")


if __name__ == "__main__":
    main()