import torch
import torch.nn as nn


class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(32, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, sentence_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(sentence_embedding)