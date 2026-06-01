import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPCaptionGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        clip_model_name="openai/clip-vit-base-patch32",
        token_embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        freeze_clip=True,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.clip_model_name = clip_model_name

        self.clip = CLIPModel.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        clip_dim = self.clip.config.projection_dim

        self.embedding = nn.Embedding(
            vocab_size,
            token_embed_dim,
            padding_idx=pad_id,
        )

        self.image_to_hidden = nn.Linear(clip_dim, hidden_dim)

        self.gru = nn.GRU(
            input_size=token_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode_image(self, pixel_values):
        if not any(p.requires_grad for p in self.clip.parameters()):
            with torch.no_grad():
                image_features = self.clip.get_image_features(
                    pixel_values=pixel_values
                )
        else:
            image_features = self.clip.get_image_features(
                pixel_values=pixel_values
            )

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-6)

        return image_features

    def forward(self, pixel_values, input_ids):
        image_features = self.encode_image(pixel_values)

        embedded = self.embedding(input_ids)

        h0 = torch.tanh(self.image_to_hidden(image_features))
        h0 = h0.unsqueeze(0)

        outputs, _ = self.gru(embedded, h0)
        logits = self.output(outputs)

        return logits

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        bos_id,
        eos_id,
        max_len=40,
        temperature=0.0,
    ):
        self.eval()

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        image_features = self.encode_image(pixel_values)

        hidden = torch.tanh(self.image_to_hidden(image_features))
        hidden = hidden.unsqueeze(0)

        cur = torch.tensor([[bos_id]], device=pixel_values.device)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(cur)
            out, hidden = self.gru(emb, hidden)
            logits = self.output(out[:, -1, :])

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(1)

            token_id = int(next_id.item())

            if token_id == eos_id:
                break

            generated.append(token_id)
            cur = next_id.view(1, 1)

        return generated


class TextOnlyGRULM(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        token_embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            token_embed_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=token_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        outputs, _ = self.gru(embedded)
        logits = self.output(outputs)
        return logits