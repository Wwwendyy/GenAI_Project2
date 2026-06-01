import argparse

import torch
from PIL import Image
from transformers import CLIPProcessor

from vocab import Vocab
from model import CLIPCaptionGRU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/clip_caption_best.pt")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vocab = Vocab.load(args.vocab)

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]

    clip_model_name = train_args.get(
        "clip_model_name",
        "openai/clip-vit-base-patch32",
    )

    processor = CLIPProcessor.from_pretrained(clip_model_name)

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

    image = Image.open(args.image).convert("RGB")
    pixel_values = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(device)

    ids = model.generate(
        pixel_values=pixel_values,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        max_len=args.max_len,
        min_len=args.min_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    caption = vocab.decode(ids)

    print("\nImage:")
    print(args.image)

    print("\nGenerated token ids:")
    print(ids)

    print("\nGenerated caption:")
    print(caption)


if __name__ == "__main__":
    main()
