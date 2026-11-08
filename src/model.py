import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcFace(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 30.0, m: float = 0.5, label_smoothing: float = 0.0):
        super().__init__()
        self.s, self.m = s, m
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor):
        cosine = F.linear(emb, F.normalize(self.weight, dim=1))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        if self.label_smoothing > 0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / cosine.size(1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    @torch.no_grad()
    def cosine_logits(self, emb: torch.Tensor) -> torch.Tensor:
        return F.linear(emb, F.normalize(self.weight, dim=1))


class IrisEmbeddingNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, dropout: float = 0.4, arcface_s: float = 30.0, arcface_m: float = 0.5, label_smoothing: float = 0.0, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.arcface = ArcFace(
            embedding_dim,
            num_classes,
            s=arcface_s,
            m=arcface_m,
            label_smoothing=label_smoothing,
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        emb = self.head(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x, labels=None):
        emb = self.embed(x)
        if labels is None:
            return emb
        loss, logits = self.arcface(emb, labels)
        return emb, loss, logits

    def set_backbone_trainable(self, flag: bool):
        for p in self.backbone.parameters():
            p.requires_grad = flag

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.arcface.cosine_logits(self.embed(x))
