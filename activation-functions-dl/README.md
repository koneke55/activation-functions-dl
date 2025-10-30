# Activation Functions for Deep Learning

A lightweight toolkit to explore, implement, and visualize popular activation functions used in modern deep learning.

- Clean, NumPy-only implementations in `src/activations.py`
- Ready-to-run visualization script in `examples/plot_activations.py`
- Colab-friendly notebook in `notebooks/activation_functions_colab.ipynb`

---

## Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Quick Plot (CLI)](#quick-plot-cli)
  - [Python API](#python-api)
  - [Notebook (Colab/Local)](#notebook-colablocal)
- [Implemented Activations](#implemented-activations)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## Features

- NumPy implementations of classic and modern activations
- Numerically stable variants for `sigmoid`, `softplus`, and `softmax`
- Example script to render a grid of activation curves
- Minimal dependencies for easy setup and portability

---

## Getting Started

### Prerequisites
- Python 3.9+

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

---

## Usage

### Quick Plot (CLI)
Render and save a grid of activation functions to `examples/activations.png`:
```powershell
python .\examples\plot_activations.py
```

### Python API
Use any activation directly in Python:
```python
import numpy as np
from pathlib import Path
import sys

# optional: make src importable without installing as a package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from activations import relu, sigmoid, softmax

x = np.linspace(-3, 3, 7)
print("x:", x)
print("ReLU:", relu(x))
print("Sigmoid:", sigmoid(x))
print("Softmax:", softmax(np.array([[1.0, 2.0, 3.0]]), axis=-1))
```

### Notebook (Colab/Local)
Open the prebuilt notebook:
- Local: `notebooks/activation_functions_colab.ipynb`
- Colab: upload the repository folder and run the first cell to adjust `sys.path`

---

## Implemented Activations
All functions operate on NumPy arrays and return NumPy arrays.

- `linear(x)`
- `sigmoid(x)` — stable formulation
- `tanh(x)`
- `relu(x)`
- `leaky_relu(x, negative_slope=0.01)`
- `elu(x, alpha=1.0)`
- `selu(x)` — with paper constants
- `softplus(x)` — stable formulation
- `softsign(x)`
- `gelu(x, approximate=True)` — tanh approximation; exact variant uses `erf`
- `swish(x, beta=1.0)`
- `mish(x)`
- `softmax(x, axis=-1)` — stable via max subtraction
- `hard_sigmoid(x)`

See `src/activations.py` for details.

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

## Development

- Style: clear and readable Python with NumPy
- Tests: you can add quick checks in the notebook or example script
- Suggestions and contributions are welcome—feel free to open an issue or PR

Recommended workflow:
```bash
# create & activate venv, then
pip install -r requirements.txt
python examples/plot_activations.py
```

---

## License

MIT License © 2025 SAMBOU KONE. See `LICENSE` for details.
