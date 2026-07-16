# Benchmarks

`archive/` preserves the named research lineage. `results/reference/` preserves compact recorded outputs used by the paper and documentation.

Start with the fast install contract:

```bash
dendritron smoke
pytest
```

Then install the relevant optional dependency group before running an archived experiment:

```bash
pip install -e ".[research]"     # NumPy/pandas/scikit-learn experiments
pip install -e ".[transformer]"  # SmolLM2/LoRA experiment
```

Historical scripts are deliberately versioned rather than silently rewritten. Some retain their original `/mnt/data` or `/content` output targets. Copy a script into a new version when changing its protocol.

