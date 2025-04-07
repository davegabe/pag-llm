import torch
import torch.nn as nn


class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(),

            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, sentence_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(sentence_embedding)
