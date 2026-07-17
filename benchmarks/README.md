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

Historical scripts are deliberately versioned rather than silently rewritten. Copy a script into a new version when changing its protocol. The SmolLM2 showcase and the notebooks still retain their original `/content` output targets.

## Running the archived experiments locally

The four `.py` sweep scripts resolve their output directory from `DENDRITRON_OUTPUT_DIR`
(default `benchmarks/results/local/`) and import cross-script dependencies relative to
their own location, so no Colab environment is required:

```bash
python benchmarks/archive/dendritron_mixed_geometry_v0_9.py
DENDRITRON_OUTPUT_DIR=/path/to/outputs python benchmarks/archive/dendritron_optdigits_benchmark.py
```

