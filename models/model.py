import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ViolenceDetector(nn.Module):
    def __init__(self, num_frames=16, num_classes=2, lstm_hidden=512,
                 dropout=0.5):
        super(ViolenceDetector, self).__init__()

        self.num_frames = num_frames

        # MobileNetV2 backbone (new weights API)
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features          # Conv layers only
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # Global Average Pooling

        # Freeze CNN
        for param in self.features.parameters():
            param.requires_grad = False

        # LSTM
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        # Classifier with dropout
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (batch, 16, 3, 224, 224)
        batch_size = x.size(0)

        # 1. Flatten frames (contiguous for memory safety)
        x = x.contiguous().view(-1, 3, 224, 224)

        # 2. Extract features with pooling
        # Frozen CNN should always be in eval mode to use stable running stats
        self.features.eval()
        with torch.no_grad():
            x = self.features(x)       # (batch×16, 1280, 7, 7)
        x = self.pool(x)           # (batch×16, 1280, 1, 1)
        x = x.flatten(1)           # (batch×16, 1280)

        # 3. Reshape for LSTM: (batch, 16, 1280)
        x = x.view(batch_size, self.num_frames, -1)

        # 4. LSTM (take last hidden state): (batch, 512)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)

        # 5. Classify with dropout: (batch, 2)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning"""
        for param in self.features.parameters():
            param.requires_grad = True


class MobileNetBaseline(nn.Module):
    def __init__(self, num_frames=16, num_classes=2):
        super().__init__()

        self.num_frames = num_frames

        backbone = mobilenet_v2(pretrained=True)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(1280, num_classes)

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)

        feats = feats.view(B, T, -1)

        video_feats = feats.mean(dim=1)

        out = self.classifier(video_feats)
        return out
