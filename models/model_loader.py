"""
models/model_loader.py

Config-driven model instantiation. A node calls get_model(model_name)
(or reads the name from its config) to get a fresh, seeded nn.Module -
no hardcoded model choice anywhere else in the pipeline.

Currently implemented: "simple_cnn" (fast, used for initial end-to-end
pipeline debugging). ResNet18, MobileNetV2, and LeNet are stubbed for
later - raising NotImplementedError until built, rather than silently
falling back to the wrong model.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Small CNN for CIFAR-10 (3x32x32 input, 10 classes). Deliberately
    lightweight - this is the debugging/warm-up model, not the final
    architecture used for headline results.

    Architecture: two conv blocks (conv -> relu -> maxpool) followed by
    two fully-connected layers.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3x32x32 -> 32x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # -> 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # -> 64x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # -> 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name: str, num_classes: int = 10, seed: int = None) -> nn.Module:
    """
    Return a freshly initialized model by config name.

    If `seed` is provided, torch's RNG is seeded immediately before
    construction so that weight initialization is reproducible - this
    matters later for fair comparisons across pipeline stages/methods
    that must start from identical initial weights.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model_name = model_name.lower()

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        raise NotImplementedError("resnet18 not yet implemented in model_loader.py")
    elif model_name == "mobilenetv2":
        raise NotImplementedError("mobilenetv2 not yet implemented in model_loader.py")
    elif model_name == "lenet":
        raise NotImplementedError("lenet not yet implemented in model_loader.py")
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'. "
                          f"Valid options: simple_cnn, resnet18, mobilenetv2, lenet")


if __name__ == "__main__":
    # Quick self-test: build the model and run a dummy CIFAR-10-shaped
    # batch through it, confirming output shape is (batch, num_classes).
    model = get_model("simple_cnn", num_classes=10, seed=42)
    dummy_batch = torch.randn(4, 3, 32, 32)  # batch of 4, CIFAR-10 shape
    output = model(dummy_batch)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape:  {tuple(dummy_batch.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (4, 10), "Output shape mismatch!"
    print("Self-test passed: output shape is (batch=4, num_classes=10)")
