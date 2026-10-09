import torch.nn as nn

from .ucihar_model import MMOFLModel, HeadEncoder


class ImageEncoder(nn.Module):
    """4-layer CNN -> 128-dim"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):  # x: (B, 3, H, W)
        return self.fc(self.conv(x).flatten(1))


class TextEncoder(nn.Module):
    """2-layer LSTM -> 128-dim, with simple word embedding"""

    def __init__(self, vocab_size=10000, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 128)

    def forward(self, x):  # x: (B, seq_len) token ids — long tensor
        _, (h, _) = self.lstm(self.embed(x))
        return self.fc(h[-1])


def build_mvsa_model(vocab_size=10000):
    """Factory: returns an MMOFLModel for MVSA-Single (image + text, 3 classes)."""
    encoders = [ImageEncoder(), TextEncoder(vocab_size=vocab_size)]
    head = HeadEncoder(feat_dim=256, num_classes=3)
    return MMOFLModel(encoders, head)

