import torch
import torch.nn as nn


class AccelEncoder(nn.Module):
    """5 conv layers + GN + 1 FC -> 128-dim"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):  # x: (B, 128, 3) -> transpose to (B, 3, 128)
        return self.fc(self.conv(x.transpose(1, 2)).squeeze(-1))


class GyroEncoder(nn.Module):
    """1 LSTM layer + 1 FC -> 128-dim"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 128, batch_first=True)
        self.fc = nn.Linear(128, 128)

    def forward(self, x):  # x: (B, 128, 3)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


class HeadEncoder(nn.Module):
    """2 FC layers, input = sum/concat of modality features"""

    def __init__(self, feat_dim=256, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, z):
        return self.net(z)


class MMOFLModel(nn.Module):
    def __init__(self, modality_encoders, head):
        super().__init__()
        self.encoders = nn.ModuleList(modality_encoders)
        self.head = head

    def encode(self, modality_data):
        """modality_data: list of tensors or None per modality"""
        features = [enc(x) for enc, x in zip(self.encoders, modality_data) if x is not None]
        return torch.cat(features, dim=-1)  # concat available features

    def forward(self, modality_data):
        return self.head(self.encode(modality_data))

    def forward_ops(self, modality_data, missing_m, proto_features):
        """OPS variant: skip encoder[missing_m], inject proto_features instead.

        proto_features: (N, feat_dim) tensor from get_proto_features() — already
        at feature level, NOT raw sensor input.  All other encoders run normally
        and receive gradients; the missing encoder is not called at all.
        """
        features = []
        for m, (enc, x) in enumerate(zip(self.encoders, modality_data)):
            if m == missing_m:
                features.append(proto_features.detach())  # no gradient through prototype
            else:
                features.append(enc(x))
        z = torch.cat(features, dim=-1)
        return self.head(z)
