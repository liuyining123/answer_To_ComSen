"""Use PyTorch autograd and gradient descent to fit a two-variable function."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


SEED = 42
OUTPUT_PATH = Path("task3_fit.png")


def target_function(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Return the noise-free function from the exercise."""
    return (
        2.0 * x1**2
        + 1.5 * x1 * x2
        + 3.0 * torch.sin(x1)
        + 0.5 * torch.cos(2.0 * x2)
        + torch.exp(-0.5 * x1)
        + 0.8 * x2
        + 2026.0
    )


class FunctionModel(torch.nn.Module):
    """The parameterized function to be fitted."""

    def __init__(self) -> None:
        super().__init__()
        self.a1 = torch.nn.Parameter(torch.tensor(1.0))
        self.a2 = torch.nn.Parameter(torch.tensor(1.0))
        self.a3 = torch.nn.Parameter(torch.tensor(1.0))
        self.a4 = torch.nn.Parameter(torch.tensor(1.0))
        self.a5 = torch.nn.Parameter(torch.tensor(-0.5))
        self.a6 = torch.nn.Parameter(torch.tensor(1.0))
        self.b = torch.nn.Parameter(torch.tensor(2000.0))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return (
            self.a1 * x1**2
            + self.a2 * x1 * x2
            + self.a3 * torch.sin(x1)
            + self.a4 * torch.cos(2.0 * x2)
            + torch.exp(self.a5 * x1)
            + self.a6 * x2
            + self.b
        )

    def parameters_as_dict(self) -> dict[str, float]:
        return {
            "a1": self.a1.item(),
            "a2": self.a2.item(),
            "a3": self.a3.item(),
            "a4": self.a4.item(),
            "a5": self.a5.item(),
            "a6": self.a6.item(),
            "b": self.b.item(),
        }


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate the coefficient of determination."""
    residual_sum = torch.sum((y_true - y_pred) ** 2)
    total_sum = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1.0 - residual_sum / total_sum).item()


def fit_model(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 6000,
) -> FunctionModel:
    model = FunctionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    for epoch in range(epochs):
        prediction = model(x1, x2)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch + 1:>4}: loss={loss.item():.6f}")

    return model


def make_plot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y_true: torch.Tensor,
    model: FunctionModel,
    r2: float,
    output_path: Path,
) -> None:
    x1_grid = torch.linspace(-3.0, 3.0, 60)
    x2_grid = torch.linspace(-3.0, 3.0, 60)
    grid_x1, grid_x2 = torch.meshgrid(x1_grid, x2_grid, indexing="ij")
    true_grid = target_function(grid_x1, grid_x2).detach().numpy()
    fit_grid = model(grid_x1, grid_x2).detach().numpy()

    x1_np = x1.detach().numpy()
    x2_np = x2.detach().numpy()
    y_true_np = y_true.detach().numpy()
    grid_x1_np = grid_x1.numpy()
    grid_x2_np = grid_x2.numpy()

    figure = plt.figure(figsize=(14, 6))
    for position, title, surface, sampled in (
        (1, "Ground Truth Function", true_grid, y_true_np),
        (2, "Fitted Function", fit_grid, model(x1, x2).detach().numpy()),
    ):
        axis = figure.add_subplot(1, 2, position, projection="3d")
        axis.plot_surface(
            grid_x1_np,
            grid_x2_np,
            surface,
            cmap="viridis" if position == 1 else "plasma",
            alpha=0.8,
            linewidth=0,
        )
        axis.scatter(x1_np, x2_np, sampled, c="red" if position == 1 else "blue",
                     s=8, label="Sampled Points")
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("y")
        axis.set_title(f"{title} ($R^2$ = {r2:.4f})")
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED)
    sample_count = 1000
    x1 = torch.empty(sample_count).uniform_(-3.0, 3.0)
    x2 = torch.empty(sample_count).uniform_(-3.0, 3.0)
    noise = torch.empty(sample_count).uniform_(-3.0, 3.0)
    y_true = target_function(x1, x2)
    y_observed = y_true + noise

    model = fit_model(x1, x2, y_observed)
    with torch.no_grad():
        fitted_observed = model(x1, x2)
        fitted_noise_free = fitted_observed.clone()

    parameters = model.parameters_as_dict()
    r2_observed = r2_score(y_observed, fitted_observed)
    r2_noise_free = r2_score(y_true, fitted_noise_free)

    print("\n拟合参数:")
    for name, value in parameters.items():
        print(f"  {name} = {value:.6f}")
    print(f"R²（含噪声样本） = {r2_observed:.6f}")
    print(f"R²（原函数，不含噪声） = {r2_noise_free:.6f}")

    make_plot(x1, x2, y_true, model, r2_noise_free, OUTPUT_PATH)
    print(f"\n原函数（不含噪声）与 PyTorch 拟合函数图像已保存到：{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()