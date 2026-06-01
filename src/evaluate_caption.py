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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/clip_caption_best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.load(os.path.join(args.data_dir, "vocab.json"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]

    clip_model_name = train_args.get(
        "clip_model_name",
        "openai/clip-vit-base-patch32",
    )

    processor = CLIPProcessor.from_pretrained(clip_model_name)

    dataset = CaptionDataset(
        os.path.join(args.data_dir, f"{args.split}.csv"),
        vocab,
        max_len=train_args.get("max_len", 40),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=make_clip_caption_collate_fn(
            processor=processor,
            pad_id=vocab.pad_id,
        ),
    )

    model = CLIPCaptionGRU(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        clip_model_name=clip_model_name,
        token_embed_dim=train_args.get("token_embed_dim", 256),
        hidden_dim=train_args.get("hidden_dim", 512),
        freeze_clip=not train_args.get("unfreeze_clip", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for pixel_values, ids, _ in tqdm(loader):
            pixel_values = pixel_values.to(device)
            ids = ids.to(device)

            input_ids = ids[:, :-1]
            target_ids = ids[:, 1:]

            logits = model(pixel_values, input_ids)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            non_pad = target_ids.ne(vocab.pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    print(f"{args.split}_loss: {avg_loss:.4f}")
    print(f"{args.split}_perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()