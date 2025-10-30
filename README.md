# Activation Functions for Deep Learning
    
An educational, hands-on repository to learn, implement, and visualize activation functions used in Deep Learning. This project is ideal for students and practitioners who want a clear, practical understanding of how activations behave, why they matter, and how to use them.

- Clean, NumPy-only implementations in `src/activations.py`
- Ready-to-run visualization in `examples/plot_activations.py`
- Colab-friendly notebook in `notebooks/activation_functions_colab.ipynb`
- Focus on numerical stability and intuition

---

## Why Activation Functions Matter
Activation functions introduce non-linearity, allowing neural networks to approximate complex functions. The choice of activation affects:
- Gradient flow (vanishing/exploding)
- Training stability and speed
- Model expressivity and performance

This repo demonstrates shapes, properties, and practical differences to build intuition.

---

## Table of Contents
- [Key Learning Outcomes](#key-learning-outcomes)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Generate Plots](#generate-plots)
  - [Use in Python](#use-in-python)
  - [Notebook (Colab/Local)](#notebook-colablocal)
- [Implemented Functions](#implemented-functions)
- [Visual Intuition](#visual-intuition)
- [Numerical Stability Notes](#numerical-stability-notes)
- [Project Structure](#project-structure)
- [References and Further Study](#references-and-further-study)
- [Contributing](#contributing)
- [License](#license)

---

## Key Learning Outcomes
- Understand the shapes and gradients of common activations
- Know when to prefer ReLU-family vs smooth activations (GELU/Swish/Mish)
- Recognize stability concerns (e.g., sigmoid, softmax, softplus) and how to address them
- Build intuition by experimenting with parameters (e.g., `leaky_relu` slope, `swish` beta)

---

## Quick Start

### Installation
PowerShell (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

bash (Linux/macOS):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Plots
```powershell
python .\examples\plot_activations.py
```
This will display a grid of activation curves and save `examples/activations.png`.

### Use in Python
```python
import numpy as np
from pathlib import Path
import sys

# allow importing from src without installing a package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from activations import relu, sigmoid, softmax

x = np.linspace(-3, 3, 7)
print("ReLU:", relu(x))
print("Sigmoid:", sigmoid(x))
print("Softmax:", softmax(np.array([[1.0, 2.0, 3.0]]), axis=-1))
```

### Notebook (Colab/Local)
- Local: open `notebooks/activation_functions_colab.ipynb`
- Colab: upload the repo folder, open the notebook, and run the first cell to set `sys.path`

---

## Implemented Functions
All functions operate on NumPy arrays and return NumPy arrays.

- `linear(x)`
- `sigmoid(x)` — stable formulation to avoid overflow/underflow
- `tanh(x)`
- `relu(x)`
- `leaky_relu(x, negative_slope=0.01)`
- `elu(x, alpha=1.0)`
- `selu(x)` — with paper constants (self-normalizing nets)
- `softplus(x)` — stable formulation
- `softsign(x)`
- `gelu(x, approximate=True)` — tanh approximation; exact variant uses `erf`
- `swish(x, beta=1.0)`
- `mish(x)`
- `softmax(x, axis=-1)` — stable via max subtraction
- `hard_sigmoid(x)`

Tip: Try toggling parameters and comparing shapes side-by-side.

---

## Visual Intuition
Below is an example grid produced by `examples/plot_activations.py`:

```
examples/activations.png
```

What to observe:
- Saturating vs non-saturating behavior (sigmoid/tanh vs ReLU/Swish/GELU)
- Smoothness and differentiability (ReLU kink vs smooth curves)
- Output range and gradient implications (e.g., tanh in [-1, 1])

---

## Numerical Stability Notes
- `sigmoid(x)`: computed using a piecewise stable form to avoid `exp(±x)` overflow
- `softplus(x)`: uses `log1p` for small values; switches formula for large `x`
- `softmax(x)`: subtracts the row-wise max to prevent overflow before exponentiation

These practices mirror production-grade deep learning libraries.

---

## Project Structure
```
activation-functions-dl/
├── notebooks/
│   └── activation_functions_colab.ipynb
├── src/
│   ├── __init__.py
│   └── activations.py
├── examples/
│   └── plot_activations.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## References and Further Study
- Goodfellow, Bengio, Courville — Deep Learning (MIT Press)
- Hendrycks & Gimpel — GELU (Gaussian Error Linear Units)
- Ramachandran, Zoph, Le — Swish: a Self-Gated Activation Function
- Misra — Mish: A Self Regularized Non-Monotonic Neural Activation Function
- PyTorch / TensorFlow docs on activation functions

---

## Contributing
Contributions are welcome! Ideas to improve learning value:
- Add derivative visualizations (first/second derivatives)
- Interactive sliders (e.g., Streamlit/Gradio) for parameters
- Comparative training demos on toy datasets

Open an issue or submit a PR.

---

## License
MIT License © 2025 SAMBOU KONE — see `LICENSE`.
