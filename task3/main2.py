"""Train a simple MLP to recognize handwritten digits stored in a Parquet file."""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
IMAGE_SIZE = 28
CLASS_COUNT = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def decode_image(value: object) -> np.ndarray:
    """Decode a Hugging Face image column value into a normalized grayscale array."""
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is None:
            raise ValueError("image 字段缺少 bytes 数据")
    elif isinstance(value, (bytes, bytearray)):
        image_bytes = bytes(value)
    else:
        raise TypeError(f"不支持的 image 数据类型: {type(value).__name__}")

    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.asarray(gray, dtype=np.float32) / 255.0


def load_dataset(path: Path, test_ratio: float, seed: int) -> tuple[TensorDataset, TensorDataset]:
    """Read images and labels with pandas, then create reproducible train/test splits."""
    frame = pd.read_parquet(path, columns=["image", "label"])
    if frame.empty:
        raise ValueError(f"数据文件为空: {path}")

    images = np.stack([decode_image(value) for value in frame["image"]])
    labels = frame["label"].to_numpy(dtype=np.int64).copy()
    features = torch.from_numpy(images.reshape(len(images), -1))
    targets = torch.from_numpy(labels)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(frame), generator=generator)
    test_size = max(1, int(len(frame) * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return (
        TensorDataset(features[train_indices], targets[train_indices]),
        TensorDataset(features[test_indices], targets[test_indices]),
    )


class DigitMLP(nn.Module):
    """A compact fully connected network for 28x28 handwritten digits."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, CLASS_COUNT),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += loss_function(logits, labels).item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train_model(
    model: DigitMLP,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    device: torch.device,
) -> dict[str, list[float]]:
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = correct / total
        _, test_acc = evaluate(model, test_loader, loss_function, device)
        history["loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(
            f"Epoch {epoch + 1:>2}/{epochs}: "
            f"Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}"
        )

    return history


def save_training_plots(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, history["loss"], marker="o", color="tab:red")
    axis.set(title="MLP Training Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    axis.plot(epochs, history["test_acc"], marker="s", label="Test Acc")
    axis.set(title="MLP Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_curve.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["loss"], color="tab:red")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[1].plot(epochs, history["train_acc"], color="tab:blue")
    axes[1].set(title="Train Acc", xlabel="Epoch")
    axes[2].plot(epochs, history["test_acc"], color="tab:green")
    axes[2].set(title="Test Acc", xlabel="Epoch")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "training_metrics.png", dpi=160)
    plt.close(figure)


def save_predictions(
    model: DigitMLP,
    dataset: TensorDataset,
    device: torch.device,
    output_path: Path,
    count: int = 9,
) -> None:
    model.eval()
    count = min(count, len(dataset))
    indices = torch.arange(count)
    inputs = dataset.tensors[0][indices].to(device)
    labels = dataset.tensors[1][indices]
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1).cpu()

    figure, axes = plt.subplots(3, 3, figsize=(7, 7))
    for axis, image, label, prediction in zip(axes.flat, inputs.cpu(), labels, predictions):
        axis.imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
        color = "green" if label.item() == prediction.item() else "red"
        axis.set_title(f"True: {label.item()}  Pred: {prediction.item()}", color=color)
        axis.axis("off")
    figure.suptitle("MLP Handwritten Digit Predictions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 PyTorch MLP 识别手写数字")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("train-00000-of-00001.parquet"),
        help="Parquet 数据集路径",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("mlp_output"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not 0.0 < args.test_ratio < 1.0:
        parser.error("--test-ratio 必须在 0 和 1 之间")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs 和 --batch-size 必须为正整数")

    set_seed(args.seed)
    device = torch.device("cpu")
    train_dataset, test_dataset = load_dataset(args.data, args.test_ratio, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = DigitMLP().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据集: {args.data}")
    print(f"训练集: {len(train_dataset)} 条, 测试集: {len(test_dataset)} 条")
    print(f"设备: {device}")
    history = train_model(model, train_loader, test_loader, args.epochs, device)
    save_training_plots(history, args.output_dir)
    save_predictions(model, test_dataset, device, args.output_dir / "predictions.png")

    print("\n最终指标:")
    print(f"Loss = {history['loss'][-1]:.6f}")
    print(f"Train Acc = {history['train_acc'][-1]:.6f}")
    print(f"Test Acc = {history['test_acc'][-1]:.6f}")
    print(f"结果图片已保存到: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
