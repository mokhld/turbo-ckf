# Examples

Runnable scripts that exercise the `turbo-ckf` Python API.

| Script | What it shows |
| ------ | ------------- |
| [`quickstart_cv.py`](quickstart_cv.py) | Constant-velocity 1-D tracking with noisy position measurements. Demonstrates the vectorized `fx` / `hx` callback contract. |

## Running

From the repo root, with the project virtualenv activated (see
[`CONTRIBUTING.md`](../CONTRIBUTING.md)):

```bash
.venv-turbo-ckf/bin/python examples/quickstart_cv.py
```

Each example is self-contained and prints to stdout. None of them write files.
