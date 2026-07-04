# DisasterM3 Repository Analysis

This report studies the actual `Junjue-Wang/DisasterM3` codebase (forked, master branch) to understand its current design and identify what would need to change to generalize it into a multi-dataset, multi-model evaluation framework.

## 1. Current Repository Structure

The repo is intentionally minimal — it ships a benchmark *runner*, not a framework:

```
DisasterM3/
├── models/
│   └── __init__.py      # ModelConfig ABC + Qwen/InternVL/Llava subclasses
├── pyscripts/
│   └── run_vllm.py       # single CLI entrypoint: loads data, runs model, dumps raw responses
├── __init__.py
└── README.md
```

Two files carry the entire logic:

- **`models/__init__.py`** defines an abstract `ModelConfig` class with one required method, `get_prompt_from_question()`, plus three concrete subclasses (`QwenVL`, `InternVL`, `Llava`) that each hardcode their own vLLM `EngineArgs` (tensor-parallel size, max context length, stop tokens) based on substring-matching the `model_id` string (e.g. `"72b" in model_id.lower()`). A factory function, `build_model_config()`, does the same string-matching to decide which subclass to instantiate.
- **`pyscripts/run_vllm.py`** is a single script that: reads a `data/<subset>.json` file, builds a per-subset prompt from a hardcoded `prompt_libs` dict (7 fixed prompt templates keyed by task name), assembles vLLM-style multimodal messages, runs generation in batches, and writes raw text responses to `results/<subset>/<model>/finished.jsonl`.

There is **no evaluation/scoring code, no configuration file, and no experiment tracking** anywhere in the repo — the benchmark stops at "generate and save the model's raw text output." Scoring against ground truth is left entirely to the user.

## 2. Code Organization Analysis

Judged against the `Dataset → Model Runner → Evaluator → Experiment Tracker` separation the internship framework is meant to have, the current code collapses everything into two files with no layer boundaries:

| Concern | Where it lives now | Problem |
|---|---|---|
| Dataset loading | Inline in `run_vllm.py`'s `__main__` block (`json.load(open(f"{subset}.json"))`) | Not a class/interface — just a file read at the bottom of the script |
| Task-specific prompting | `prompt_libs` dict + `if subset in [...]` branches in `get_messages_from_data()` | Adding a task means editing this function's branching logic, not adding a new file |
| Model selection & config | `build_model_config()` string-matching on `model_id` | Adding a model means editing the factory's `if/elif` chain, not registering a plugin |
| Execution loop | Hardcoded in `__main__` | Batch size, resume-from-checkpoint, and generation logic are not reusable outside this script |
| Evaluation | **Absent** | No metrics computation exists; only raw responses are saved |
| Experiment tracking | **Absent** | No run metadata, no MLflow/W&B integration |
| Configuration | CLI flags only (`argparse`) | No YAML/config file; every run is fully specified on the command line |

The one genuine abstraction present is `ModelConfig` (an `ABC` with `get_prompt_from_question()` as an abstract method) — this is a real, reusable pattern and is the closest thing to the "Model Runner" layer the target architecture calls for.

## 3. Is the Framework Tied to a Specific Dataset?

**Yes, tightly.** Several assumptions are baked directly into `run_vllm.py` and are not configurable:

- **Fixed file layout:** data is expected at `f"{PROJECT_ROOT}/data/{subset}.json"` with images under `f"{PROJECT_ROOT}/data/images/"`. `PROJECT_ROOT` is derived by walking up from the script's own file path — so the "dataset" is really "whatever JSON happens to be sitting in this repo's `data/` folder," not a pluggable source.
- **Fixed task vocabulary:** `get_messages_from_data()` only recognizes six hardcoded subset names (`bearing_body`, `building_damage_counting`, `disaster_type`, `road_damage_counting`, `landuse`, `relational_reasoning_qa`, plus `caption`/`recovery`). A new dataset with different task names would need this function edited directly.
- **Fixed field names:** the code expects specific JSON keys per subset (`pre_image_path`, `post_image_path`, `prompts`, `options_str`, and inconsistently `option_str` for one subset — a naming bug in the current code). A dataset like MONITRS or EarthVQA, with different field names or a different bi-temporal/single-image structure, would not load without rewriting this function.

**What it would take to use another dataset today:** a developer would need to either (a) preprocess the new dataset into DisasterM3's exact JSON schema and file layout, or (b) fork `get_messages_from_data()` and add another `elif` branch — both defeat the point of a reusable framework.

## 4. Proposed Modular Redesign

To decouple dataset, model, and evaluation logic (per the target `datasets/`, `models/`, `evaluation/`, `experiments/` structure):

**Dataset layer** — introduce a `BaseDataset` interface (`load()`, `__getitem__`) so each dataset (`disasterm3.py`, `monitrs.py`, `earthvqa.py`) is responsible only for mapping its own on-disk format into a common record shape (e.g. `{images: [...], prompt: str, task: str, reference: ...}`). This isolates the six hardcoded `if subset in [...]` branches into per-dataset adapter classes.

**Model layer** — keep and extend the existing `ModelConfig` ABC pattern (it already works well), but replace the string-matching factory with a registry (e.g. a dict mapping model-family name → class, or entry-point-based plugin discovery) so adding `internvl_runner.py` or `qwen_runner.py` doesn't require editing a shared `if/elif` chain.

**Evaluation layer** — this is entirely new, since none exists today. Task-specific evaluators (`vqa.py` for accuracy/F1 on choice-based tasks, `damage_assessment.py` for counting/classification metrics) would consume the same `finished.jsonl`-style output the current script already produces, plus the dataset's ground-truth references, and output metrics rather than requiring manual post-hoc scoring.

**Configuration & experiment tracking** — replace `argparse`-only invocation with YAML configs specifying `dataset`, `model`, `task`, and `tracker` sections, and wrap the existing generation loop with an experiment tracker (MLflow/W&B) call that logs the same metadata (`model_id`, `subset`, batch size, results path) already present as local variables in `run_vllm.py` today — the values exist, they're just never persisted anywhere.

**What's reused as-is:** the `ModelConfig` abstract base class and its three subclasses, the vLLM-based batched generation loop, and the `finished.jsonl` resume-from-checkpoint logic all transfer directly into the new framework with minimal change — they're already reasonably decoupled from the dataset-specific logic that surrounds them.
