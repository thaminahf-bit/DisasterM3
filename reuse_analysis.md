# Reuse Analysis: EarthVQA

This note analyzes one reusable design pattern from `Junjue-Wang/EarthVQA` (cloned and inspected directly, not from documentation alone) and how it addresses a gap identified in `analysis.md` for DisasterM3.

## The Pattern: Registry + Config-Driven Instantiation

EarthVQA's config file (`configs/earthvqa.py`) specifies components like this:

```python
data = dict(
    train=dict(
        type='EarthVQALoader',
        params=dict(
            qa_path='./EarthVQA/Train_QA.json',
            batch_size=16,
            ...
        ),
    ),
)
```

The actual class is defined in `data/earthvqa.py` as:

```python
@er.registry.DATALOADER.register()
class EarthVQALoader(DataLoader, ConfigurableMixin):
    ...
```

The `@er.registry.DATALOADER.register()` decorator adds the class to a global lookup table keyed by its own class name. At runtime, the framework reads the config's `type` string (`'EarthVQALoader'`), looks it up in the registry, and instantiates it with `params` as keyword arguments — no `if/elif` chain matching class names anywhere in the codebase.

## Why This Is Reusable

This directly solves a coupling problem documented in `analysis.md` for DisasterM3: that repo's `build_model_config()` picks a model class via hardcoded string-matching (`if "qwen" in model_id.lower()`), and adding a dataset or model means editing a shared factory function. EarthVQA's registry pattern removes that coupling entirely — adding a new dataset or model means writing a new class with a decorator on it, full stop; no existing file needs to be touched or even known about by whoever adds the new component.

## How It Fits the Proposed Framework

The internship brief's target architecture calls for "no hardcoded dataset/model selection" via YAML config — this pattern is a proven, working implementation of exactly that requirement, adaptable to both the `datasets/` and `models/` layers:

- **`datasets/base.py`** — add a small registry (a dict mapping name → class, populated by a `@register_dataset("disasterm3")` decorator) instead of the caller needing to `import DisasterM3Dataset` by name.
- **`models/base.py`** — same pattern replaces `build_model_config()`'s string-matching with `@register_model("qwen_vl")`, `@register_model("internvl")`, etc.
- **`configs/*.yaml`** — mirrors EarthVQA's `type` + `params` structure, e.g.:
  ```yaml
  dataset:
    type: disasterm3
    params:
      data_root: ./data
      subset: bearing_body
  model:
    type: qwen_vl
    params:
      model_id: Qwen/Qwen2.5-VL-7B-Instruct
  ```

This keeps the registry lightweight (a plain dict + decorator, no need to depend on the full `ever` library) while adopting the same core idea: **the config file names a component by string; a registry resolves the string to a class; nothing in the framework core needs to know about specific datasets or models in advance.** This is the missing piece that turns the current `datasets/base.py` / `models/__init__.py` abstractions (which already exist as classes) into something actually swappable purely through configuration, which is the framework's central requirement.
