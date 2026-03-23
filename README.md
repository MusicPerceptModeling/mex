# ic_benchmark_suite

Installable package for IC benchmark tasks, including repetition-vs-IC analysis on SALAMI-style segment annotations. Consistency with IDyOM (pitch surprisal task). 

## Installation

Core package:

```bash
pip install .
```

Baselines require PyTorch, but PyTorch should be installed manually to match the local system
(CPU/CUDA/ROCm/MPS, driver, and platform).

1. Install PyTorch with the official selector: https://pytorch.org/get-started/locally/
2. Install baseline extras:

```bash
pip install ".[baselines]"
```