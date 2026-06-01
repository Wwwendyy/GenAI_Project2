import os
import argparse
import random
import pandas as pd
from vocab import Vocab


def find_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def resolve_image_path(image_value, image_root):
    image_value = str(image_value).strip()

    candidates = []

    if os.path.isabs(image_value):
        candidates.append(image_value)
    else:
        candidates.append(os.path.join(image_root, image_value))

    more_candidates = []
    for path in candidates:
        more_candidates.append(path)

        root, ext = os.path.splitext(path)
        if ext == "":
            more_candidates.append(path + ".jpg")
            more_candidates.append(path + ".jpeg")
            more_candidates.append(path + ".png")

    for path in more_candidates:
        if os.path.exists(path):
            return path

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--image-col", type=str, default=None)
    parser.add_argument("--caption-col", type=str, default=None)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    print("CSV columns:", list(df.columns))

    image_col = args.image_col or find_column(
        df.columns,
        ["image", "image_path", "filename", "file_name", "img", "path", "image file"],
    )
    caption_col = args.caption_col or find_column(
        df.columns,
        ["caption", "text", "description", "comment"],
    )

    if image_col is None or caption_col is None:
        raise ValueError(
            "Could not detect image/caption columns. "
            "Please pass --image-col and --caption-col manually."
        )

    print(f"Using image column: {image_col}")
    print(f"Using caption column: {caption_col}")

    rows = []
    missing_images = 0
    missing_captions = 0

    for _, row in df.iterrows():
        caption = row[caption_col]

        if pd.isna(caption) or len(str(caption).strip()) == 0:
            missing_captions += 1
            continue

        image_path = resolve_image_path(row[image_col], args.image_root)

        if image_path is None:
            missing_images += 1
            continue

        rows.append({
            "image_path": image_path,
            "caption": str(caption).strip()
        })

    print(f"Valid image-caption pairs: {len(rows)}")
    print(f"Skipped missing captions: {missing_captions}")
    print(f"Skipped missing images: {missing_images}")

    if len(rows) == 0:
        raise ValueError("No valid image-caption pairs found. Check image-root and CSV columns.")

    random.seed(args.seed)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    pd.DataFrame(train_rows).to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    vocab = Vocab(min_freq=args.min_freq, max_size=args.max_vocab_size)
    vocab.build([r["caption"] for r in train_rows])
    vocab.save(os.path.join(args.out_dir, "vocab.json"))

    print(f"Train: {len(train_rows)}")
    print(f"Val:   {len(val_rows)}")
    print(f"Test:  {len(test_rows)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Saved processed data to {args.out_dir}")


if __name__ == "__main__":
    main()
