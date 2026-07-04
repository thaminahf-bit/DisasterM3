# Execution Notes — Attempting to Run DisasterM3's `run_vllm.py`

This documents a real attempt to follow the README and execute the benchmark script, including every issue actually encountered (not hypothetical ones).

## Environment
- Python 3.12.3, Linux (CPU-only sandbox — **no GPU available**, ~3.9 GB RAM, limited disk quota)
- No `requirements.txt` or `pyproject.toml` exists in the repo — dependencies had to be inferred entirely from the `import` statements at the top of `models/__init__.py` and `pyscripts/run_vllm.py`.

## Dependencies (inferred from source, since none are declared)
From reading the imports directly:
`torch`, `torchvision`, `transformers`, `vllm`, `Pillow`, `tqdm`, `qwen_vl_utils`, `decord` (wrapped in a `try/except ModuleNotFoundError`, so effectively optional).

## Step 1 — Follow the README literally
The README's exact command is:
```
python disaster_m3/pyscripts/run_vllm.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --subset bearing_body
```
**Result:** fails immediately.
```
python3: can't open file '.../DisasterM3/disaster_m3/pyscripts/run_vllm.py': [Errno 2] No such file or directory
```
**Issue found:** the README assumes the repo is nested inside a `disaster_m3/` folder, but the actual repo root already contains `pyscripts/` directly — there is no `disaster_m3/` subfolder. This is a path bug in the README, not a local setup mistake. → **Fixed in README** (see below).

## Step 2 — Correct the path and retry
```
python3 pyscripts/run_vllm.py --help
```
**Result:** fails before argparse even runs:
```
ModuleNotFoundError: No module named 'vllm'
```
This confirms all heavy dependencies are imported at module load time, before any CLI parsing — so even `--help` requires the full dependency stack to be installed first.

## Step 3 — Install dependencies incrementally
Installed the lighter dependencies first to isolate failures one at a time:
```
pip install transformers qwen_vl_utils torchvision tqdm
```
These installed cleanly (`torch` came in automatically as a dependency of `torchvision`). With these in place:
```python
from models import build_model_config, ModelConfig   # succeeds
```
The `models/__init__.py` module — the one genuine abstraction in the repo — imports and works fine on CPU with no GPU-specific dependencies of its own.

## Step 4 — Install `vllm`
```
pip install vllm
```
**Result:** failed —
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```
`vllm` pulls in a large stack of CUDA-specific packages (`nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-cusolver`, `nvidia-nccl-cu13`, etc. — 15 separate multi-hundred-MB packages), even when the intent is CPU-only use. On this environment's disk quota, the install exhausted available space partway through.

**Root cause, not just a local quirk:** `vllm` is a GPU-first inference engine. It is not meant to run meaningfully without CUDA hardware — even if the install had succeeded, `LLM(**engine_args)` in `run_vllm.py` would still require an actual GPU to load a 7B+ parameter VLM. This is a genuine hardware requirement of the benchmark, not a fixable dependency issue.

## Step 5 — Dataset access
Independent of the code, the dataset itself (`data/*.json` + `data/images/`) is **not bundled in the repo** and is gated behind a Google Form (linked in the README) rather than a direct download. Even with a working GPU environment, a full run requires waiting on manual dataset access approval.

## Summary of Encountered Issues

| # | Issue | Category |
|---|---|---|
| 1 | README's example path (`disaster_m3/pyscripts/...`) doesn't match actual repo layout | Documentation bug — fixed in README |
| 2 | No `requirements.txt` — dependencies must be reverse-engineered from imports | Reproducibility gap |
| 3 | All heavy dependencies import at module load, so even `--help` needs the full stack installed | Design limitation |
| 4 | `vllm` install requires several GB of CUDA-specific packages, infeasible on CPU-only/low-disk environments | Hardware requirement, not a bug |
| 5 | Actual model inference requires a real GPU — CPU-only execution is not viable regardless of install success | Hardware requirement |
| 6 | Dataset is gated behind a Google Form, not bundled or directly downloadable | Data access limitation |
| 7 | README's second example uses `--subset report`, but `report` does not appear as a key in `prompt_libs`, nor as a recognized subset in `get_messages_from_data()` — this example would raise `ValueError('Unknown subset report')` even with a working GPU setup | Documentation/code inconsistency (not fixed — behavior change out of scope for this task) |

## What Was Actually Verified to Work
- The `models/` package (the `ModelConfig` abstraction and its `QwenVL`/`InternVL`/`Llava` subclasses) imports and constructs cleanly on CPU with no GPU-specific setup — this module is genuinely portable.
- The script's control flow up to the `LLM(**engine_args)` call (argument parsing, dataset JSON loading logic, resume-from-checkpoint logic) is inspectable and understandable by static reading, even though it can't be executed end-to-end here.

## Conclusion
A full run of `run_vllm.py` requires: (1) a CUDA-capable GPU with sufficient VRAM for a 7B+ parameter VLM, (2) several GB of free disk for `vllm`'s CUDA dependency stack, and (3) approved access to the gated dataset. None of these were available in this environment, so this counts as a **partial/static execution attempt** — the failure points themselves are the useful output, since they reveal exactly where the current codebase assumes a specific (GPU-equipped, dataset-in-hand) environment rather than degrading gracefully.
