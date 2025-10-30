import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Allow running the example without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from activations import (
    linear,
    sigmoid,
    tanh,
    relu,
    leaky_relu,
    elu,
    selu,
    softplus,
    softsign,
    gelu,
    swish,
    mish,
    hard_sigmoid,
)


def main() -> None:
    x = np.linspace(-6, 6, 1000)

    activations = [
        ("linear", linear(x)),
        ("sigmoid", sigmoid(x)),
        ("tanh", tanh(x)),
        ("relu", relu(x)),
        ("leaky_relu", leaky_relu(x)),
        ("elu", elu(x)),
        ("selu", selu(x)),
        ("softplus", softplus(x)),
        ("softsign", softsign(x)),
        ("gelu", gelu(x)),
        ("swish", swish(x)),
        ("mish", mish(x)),
        ("hard_sigmoid", hard_sigmoid(x)),
    ]

    n = len(activations)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True)
    axes = axes.ravel()

    for idx, (name, y) in enumerate(activations):
        ax = axes[idx]
        ax.plot(x, y, label=name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide any unused subplots
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle("Activation Functions")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = Path(__file__).with_name("activations.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()


