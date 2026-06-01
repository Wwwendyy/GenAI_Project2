import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class CaptionDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len=40):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        caption = row["caption"]

        image = Image.open(image_path).convert("RGB")

        ids = self.vocab.encode(caption, max_len=self.max_len)
        ids = torch.tensor(ids, dtype=torch.long)

        return image, ids, caption


class TextOnlyDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len=40):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.iloc[idx]["caption"]
        ids = self.vocab.encode(caption, max_len=self.max_len)
        ids = torch.tensor(ids, dtype=torch.long)
        return ids, caption


def make_clip_caption_collate_fn(processor, pad_id):
    def collate_fn(batch):
        images, ids_list, captions = zip(*batch)

        pixel_values = processor(
            images=list(images),
            return_tensors="pt"
        )["pixel_values"]

        max_len = max(len(ids) for ids in ids_list)

        padded = torch.full(
            (len(ids_list), max_len),
            fill_value=pad_id,
            dtype=torch.long,
        )

        for i, ids in enumerate(ids_list):
            padded[i, :len(ids)] = ids

        return pixel_values, padded, captions

    return collate_fn


def text_collate_fn(batch, pad_id):
    ids_list, captions = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)

    padded = torch.full(
        (len(ids_list), max_len),
        fill_value=pad_id,
        dtype=torch.long,
    )

    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids

    return padded, captions