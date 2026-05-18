"""
Description: Implementing multiple gradient descent for multi-variate linear regression.
Date: May 18, 2026
Author: Aleksa Zatezalo
"""

import copy, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
 
# Core Data  
X_train = np.array([[2104, 5, 1, 45],
                     [1416, 3, 2, 40],
                     [852,  2, 1, 35]])
y_train = np.array([460, 232, 178])
 
FEATURE_NAMES = ["sq ft", "bedrooms", "floors", "age"]

# Core Functions
def predict(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """ŷ = w · x + b  (single example)."""
    return np.dot(x, w) + b


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """MSE cost  J = (1/2m) Σ (ŷ_i - y_i)²."""
    m = X.shape[0]
    predictions = np.dot(X, w) + b       # was: X @ w + b
    errors = predictions - y
    return np.dot(errors, errors) / (2 * m)


def compute_gradient(X: np.ndarray, y: np.ndarray,
                     w: np.ndarray, b: float) -> tuple[float, np.ndarray]:
    """∂J/∂b  and  ∂J/∂w  for all features simultaneously."""
    m = X.shape[0]
    errors = np.dot(X, w) + b - y        # was: X @ w + b - y
    dj_dw = np.dot(X.T, errors) / m     # was: (X.T @ errors) / m
    dj_db = errors.mean()
    return dj_db, dj_dw


def gradient_descent(X: np.ndarray, y: np.ndarray,
                     w_init: np.ndarray, b_init: float,
                     alpha: float, num_iters: int) -> tuple:
    """
    Batch gradient descent for multivariate linear regression.

    Returns
    -------
    w          : final weight vector  (n,)
    b          : final bias           scalar
    J_history  : cost at every iteration
    w_history  : weight snapshots every 1% of training (for plotting)
    """
    w = copy.deepcopy(w_init)
    b = b_init
    J_history = []
    w_history = []
    snapshot_interval = max(1, num_iters // 100)

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % snapshot_interval == 0:
            w_history.append((i, w.copy()))

        if i % max(1, math.ceil(num_iters / 10)) == 0:
            print(f"  Iter {i:5d} | Cost {cost:12.4f}")

    return w, b, J_history, w_history

# Vizualization
def plot_results(J_history: list, w_history: list,
                 w_final: np.ndarray, b_final: float,
                 X: np.ndarray, y: np.ndarray) -> None:
    """Four-panel diagnostic dashboard."""
 
    iters = np.arange(len(J_history))
    snap_iters = [s[0] for s in w_history]
    snap_weights = np.array([s[1] for s in w_history])   # (snapshots, n)
    n_features = X.shape[1]
 
    fig = plt.figure(figsize=(14, 10), facecolor="#0f0f0f")
    fig.suptitle("Multivariate Gradient Descent — Diagnostics",
                 fontsize=16, color="white", y=0.98,
                 fontfamily="monospace")
 
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ACCENT   = "#00e5ff"
    MUTED    = "#546e7a"
    GRID_COL = "#1e1e1e"
    COLORS   = ["#00e5ff", "#ff4081", "#76ff03", "#ffea00"]   # one per feature
 
    ax_style = dict(facecolor="#151515",
                    tick_params=dict(colors="white", labelsize=8))
 
def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor("#151515")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_title(title, color="white", fontsize=10,
                    fontfamily="monospace", pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.spines[:].set_color("#2a2a2a")
    ax.grid(color=GRID_COL, linewidth=0.5)
 
    # ── Panel 1: Cost curve ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iters, J_history, color=ACCENT, linewidth=1.5, label="J(w,b)")
    ax1.set_yscale("log")
    _style(ax1, "Cost vs Iteration (log scale)", "Iteration", "Cost J")
    ax1.legend(fontsize=8, labelcolor="white", framealpha=0.2)
 
    # ── Panel 2: Weight trajectories ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for j in range(n_features):
        ax2.plot(snap_iters, snap_weights[:, j],
                 color=COLORS[j % len(COLORS)], linewidth=1.5,
                 label=FEATURE_NAMES[j])
    _style(ax2, "Weight Trajectories", "Iteration", "w value")
    ax2.legend(fontsize=7, labelcolor="white", framealpha=0.2, ncol=2)
 
    # ── Panel 3: Predicted vs Actual ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    y_pred = X @ w_final + b_final
    house_labels = [f"House {i+1}" for i in range(len(y))]
    x_pos = np.arange(len(y))
    bar_w = 0.35
    ax3.bar(x_pos - bar_w/2, y,      bar_w, label="Actual",    color=ACCENT,  alpha=0.85)
    ax3.bar(x_pos + bar_w/2, y_pred, bar_w, label="Predicted", color="#ff4081", alpha=0.85)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(house_labels, color="white", fontsize=8)
    _style(ax3, "Predicted vs Actual Prices ($k)", "House", "Price ($k)")
    ax3.legend(fontsize=8, labelcolor="white", framealpha=0.2)
    # annotate error
    for i, (ya, yp) in enumerate(zip(y, y_pred)):
        err = yp - ya
        ax3.annotate(f"Δ{err:+.1f}", xy=(i + bar_w/2, yp),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=7,
                     color="white", alpha=0.7)
 
    # ── Panel 4: Final weights bar chart ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.barh(FEATURE_NAMES, w_final,
                    color=[COLORS[j] for j in range(n_features)], alpha=0.85)
    ax4.axvline(0, color="white", linewidth=0.8, alpha=0.4)
    _style(ax4, "Final Weights", "Weight value", "Feature")
    ax4.tick_params(axis="y", colors="white", labelsize=8)
    for bar, val in zip(bars, w_final):
        ax4.text(bar.get_width() + (0.002 * abs(w_final).max() or 0.001),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=7, color="white")
 
    plt.savefig("/mnt/user-data/outputs/gd_diagnostics.png",
                dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print("\nPlot saved → gd_diagnostics.png")
    plt.show()
  
def main():
    n = X_train.shape[1]
    w_init = np.zeros(n)
    b_init = 0.0
    alpha     = 5.0e-7
    num_iters = 1000
 
    print("Running gradient descent …")
    w_final, b_final, J_history, w_history = gradient_descent(
        X_train, y_train, w_init, b_init, alpha, num_iters
    )
 
    print(f"\nFinal  w = {w_final}")
    print(f"Final  b = {b_final:.4f}")
    print(f"Final cost = {J_history[-1]:.4f}\n")
 
    for i, x in enumerate(X_train):
        print(f"  House {i+1}: predicted ${predict(x, w_final, b_final):.1f}k  "
              f"| actual ${y_train[i]}k")
 
    plot_results(J_history, w_history, w_final, b_final, X_train, y_train)
 
 
if __name__ == "__main__":
    main()
