"""
Training script for the audio spoofing classifier head.

Expects a CSV manifest with columns: filepath, label
  - label = 1 for spoofed/AI-generated voice, 0 for real/bona fide voice
  (This matches the ASVspoof protocol format - see README for how to build
  this manifest from the ASVspoof 2019/2021 dataset files.)

Usage:
    python train.py --manifest data/train_manifest.csv --epochs 5 --out checkpoints/audio_model.pt
"""

import argparse
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from model import AudioSpoofClassifier


class SpoofDataset(Dataset):
    def __init__(self, manifest_csv: str):
        self.df = pd.read_csv(manifest_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = sf.read(row["filepath"])
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform, sr, float(row["label"])


def collate_batch(batch, model: AudioSpoofClassifier):
    """Pads a batch of variable-length audio clips and preprocesses them together."""
    waveforms, srs, labels = zip(*batch)
    # Assumes all clips share the same sample rate (standard for ASVspoof: 16kHz)
    input_values = model.preprocess_audio(list(waveforms), srs[0])
    labels = torch.tensor(labels, dtype=torch.float32)
    return input_values, labels


def train(manifest_path: str, epochs: int, out_path: str, lr: float = 1e-4, batch_size: int = 8):
    model = AudioSpoofClassifier(freeze_encoder=True)
    dataset = SpoofDataset(manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, model),
    )

    # Only the classifier head has requires_grad=True, so this optimiser
    # automatically trains just that part.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = torch.nn.BCELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for input_values, labels in loader:
            input_values, labels = input_values.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(input_values)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"Saved trained model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with filepath,label columns")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="checkpoints/audio_model.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    train(args.manifest, args.epochs, args.out, args.lr, args.batch_size)
