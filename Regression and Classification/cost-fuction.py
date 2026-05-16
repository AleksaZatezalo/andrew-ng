"""
Description: Developing a cost function for a linear regression.
Date: May 16, 2026
Author: Aleksa Zatezalo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ── Training data ──────────────────────────────────────────────────────────────
x_train = np.array([1.0, 2.0])   # Size of house in 1000 sq ft
y_train = np.array([300.0, 500.0])  # Price in $1000s


# ── Cost function ──────────────────────────────────────────────────────────────
def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """Mean squared error cost for linear regression f(x) = wx + b."""
    m = x.shape[0]
    predictions = w * x + b          # vectorized: no explicit loop needed
    squared_errors = (predictions - y) ** 2
    return np.sum(squared_errors) / (2 * m)


# ── Parameter grid ─────────────────────────────────────────────────────────────
def build_cost_surface(
    x: np.ndarray,
    y: np.ndarray,
    w_range: tuple[float, float],
    b_range: tuple[float, float],
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate cost over a 2-D grid of (w, b) values."""
    w_vals = np.linspace(*w_range, resolution)
    b_vals = np.linspace(*b_range, resolution)
    W, B = np.meshgrid(w_vals, b_vals)
    J = np.vectorize(lambda w, b: compute_cost(x, y, w, b))(W, B)
    return W, B, J


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_cost_surface(
    W: np.ndarray,
    B: np.ndarray,
    J: np.ndarray,
) -> None:
    """Render a contour map and a 3-D surface side-by-side."""
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Linear Regression Cost  J(w, b)", fontsize=14)

    # — Contour plot ————————————————————————————————————————————————————————
    ax1 = fig.add_subplot(1, 2, 1)
    cp = ax1.contourf(W, B, J, levels=40, cmap="viridis")
    fig.colorbar(cp, ax=ax1, label="J(w, b)")
    ax1.contour(W, B, J, levels=40, colors="white", linewidths=0.4, alpha=0.3)
    ax1.set_xlabel("w")
    ax1.set_ylabel("b")
    ax1.set_title("Contour map")

    # Mark the minimum
    min_idx = np.unravel_index(np.argmin(J), J.shape)
    ax1.plot(W[min_idx], B[min_idx], "r*", markersize=12, label="minimum")
    ax1.legend()

    # — 3-D surface ——————————————————————————————————————————————————————————
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(W, B, J, cmap="viridis", alpha=0.85, linewidth=0)
    ax2.set_xlabel("w")
    ax2.set_ylabel("b")
    ax2.set_zlabel("J(w, b)")
    ax2.set_title("Surface plot")

    plt.tight_layout()
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    W, B, J = build_cost_surface(
        x_train, y_train,
        w_range=(0, 300),
        b_range=(-200, 200),
    )
    plot_cost_surface(W, B, J)