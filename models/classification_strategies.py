from abc import ABC, abstractmethod

import torch


class ClassificationStrategy(ABC):
    @staticmethod
    def from_name(name: str) -> "ClassificationStrategy":
        if isinstance(name, ClassificationStrategy):
            return name  # Directly return if already an instance
        elif name == 'grad-value':
            return GradValueClassificationStrategy()
        elif name == 'grad-subtract':
            return GradSubtractClassificationStrategy()
        elif name == 'grad-addition':
            return GradAdditionClassificationStrategy()
        else:
            raise ValueError(f"Unknown classification strategy: {name}")

    @abstractmethod
    def __call__(self, x_embed: torch.Tensor, grad_x_embed: torch.Tensor) -> torch.Tensor:
        """
        Adapt the input x_embed and grad_x_embed to the classification strategy.
        It will return the tensor to be used for classification.

        Args:
            x_embed (torch.Tensor): The input embeddings.
            grad_x_embed (torch.Tensor): The gradient embeddings.
        Returns:
            torch.Tensor: The tensor to be used for classification.
        """
        pass


class GradValueClassificationStrategy(ClassificationStrategy):
    def __call__(self, x_embed: torch.Tensor, grad_x_embed: torch.Tensor) -> torch.Tensor:
        """
        Classify using the gradient values directly, ignoring the input embeddings.
        """
        return grad_x_embed


class GradSubtractClassificationStrategy(ClassificationStrategy):
    def __call__(self, x_embed: torch.Tensor, grad_x_embed: torch.Tensor) -> torch.Tensor:
        """
        Classify by subtracting the gradient embeddings from the input embeddings.
        """
        return x_embed - grad_x_embed


class GradAdditionClassificationStrategy(ClassificationStrategy):
    def __call__(self, x_embed: torch.Tensor, grad_x_embed: torch.Tensor) -> torch.Tensor:
        """
        Classify by adding the gradient embeddings to the input embeddings.
        """
        return x_embed + grad_x_embed
