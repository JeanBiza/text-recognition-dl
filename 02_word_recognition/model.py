import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, rnn_hidden: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)

        batch, channels, height, width = features.shape
        assert height == 1, f"Expected height=1 after CNN, got {height}"
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)

        rnn_out, _ = self.rnn(features)
        logits = self.classifier(rnn_out)

        return logits


if __name__ == "__main__":
    num_classes = 62 + 1
    model = CRNN(num_classes=num_classes)
    dummy = torch.randn(4, 1, 32, 128)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  (expected: [4, 32, {num_classes}])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")