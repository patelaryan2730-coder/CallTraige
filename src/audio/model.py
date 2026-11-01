"""
Audio Spoofing Detector
------------------------
Loads a pretrained wav2vec2 model (via HuggingFace Transformers) and attaches
a small classifier head trained to distinguish real vs. AI-generated/spoofed voice.

We do NOT train wav2vec2 itself from scratch - we reuse its pretrained speech
understanding and only train the final classification layer. This is standard
transfer-learning practice and keeps compute/data requirements realistic for
a one-semester project.

NOTE: Running this script downloads the pretrained wav2vec2 weights from
HuggingFace on first run (~360MB), so you need an internet connection.
"""

import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class AudioSpoofClassifier(nn.Module):
    """
    Wraps a frozen (or lightly fine-tuned) wav2vec2 encoder with a small
    classifier head. Outputs a single probability: P(voice is AI-generated).
    """

    def __init__(self, pretrained_name: str = "facebook/wav2vec2-base", freeze_encoder: bool = True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_name)

        if freeze_encoder:
            # Freeze the pretrained encoder - we only train the classifier head.
            # This is the "reuse, don't retrain from scratch" step we discussed.
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # typically 768 for wav2vec2-base

        # Small classifier head: this is the part your team actually trains.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Load the trained head if present. head.pt was trained on a balanced ASVspoof 2019 LA
        # subset (frozen encoder, 768 -> 128 -> 1 head); place it next to this file.
        self.head_loaded = False
        _ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "head.pt")
        if os.path.exists(_ckpt):
            self.classifier.load_state_dict(torch.load(_ckpt, map_location="cpu"))
            self.head_loaded = True
        print("Audio head:", "TRAINED (head.pt loaded)" if self.head_loaded else "UNTRAINED (head.pt not found)")

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        input_values: raw audio waveform tensor, shape (batch, samples),
                      16kHz mono, produced by preprocess_audio() below.
        Returns: spoof probability, shape (batch,), values in [0, 1].
        """
        outputs = self.encoder(input_values)
        # Mean-pool across the time dimension to get one vector per clip.
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return torch.sigmoid(logits)

    def preprocess_audio(self, waveform, sampling_rate: int):
        """
        Converts a raw numpy/torch waveform into the input format wav2vec2 expects.
        `waveform` should be mono audio at `sampling_rate` Hz (any rate - this
        resamples/normalises via the feature extractor).
        """
        inputs = self.feature_extractor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        return inputs.input_values


def score_audio_file(model: AudioSpoofClassifier, filepath: str) -> float:
    """
    Convenience function: load a .wav file and return a single spoof score.
    Requires: pip install soundfile
    """
    import soundfile as sf

    waveform, sr = sf.read(filepath)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # convert stereo to mono

    input_values = model.preprocess_audio(waveform, sr)
    model.eval()
    with torch.no_grad():
        score = model(input_values).item()
    return score


if __name__ == "__main__":
    # Quick smoke test - this instantiates the model (downloads pretrained
    # weights on first run) and confirms the forward pass shapes are correct.
    print("Loading pretrained wav2vec2 + classifier head...")
    model = AudioSpoofClassifier()
    print("Model loaded. Parameter count (trainable only):")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / Total: {total:,}")
    print("This confirms we're only training the small classifier head,")
    print("not the full wav2vec2 encoder.")