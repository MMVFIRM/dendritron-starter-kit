# Experiment Lineage

The archive preserves the development sequence. Use the lightweight package for APIs and invariant tests; use these scripts when reproducing a specific scientific result.

| Version | Artifact | Question |
| --- | --- | --- |
| v0.1 | `dendritron_minsky_benchmark_v0_1.py` | Can a locally nonlinear primitive construct XOR, parity, and connectivity systems? |
| v0.2 | `dendritron_minsky_benchmark_v0_2.py` | Can learned/compiled local units establish Boolean completeness, ownership, and zero-interference assembly? |
| v0.3 | `dendritron_tissue_benchmark_v0_3.py` | Can continuous regions and a semantic router assemble into a tissue? |
| v0.4 | notebooks and README | Can immutable sharing plus copy-on-write preserve ownership? |
| v0.5 | Tissue Compiler notebook | Can exact shared factors be discovered automatically? |
| v0.7 | `dendritron_plasticity_benchmark_v0_7.py` | Can the tissue grow, split, merge, retire, recur, detect damage, and repair locally? |
| v0.8 | `dendritron_plasticity_limit_sweep_v0_8.py` | Where do plasticity and geometry controls fail? |
| v0.9 | `dendritron_mixed_geometry_v0_9.py` | Can explicit owners use compartmentalized Euclidean and hyperbolic charts? |
| LVQ | `dendritron_optdigits_benchmark.py` | Can a class-incremental, no-task-ID prototype realization retain local class owners? |
| v0.4.2 Transformer | `dendritron_smollm2_360m_showcase_v4_2_FINAL.py` | Can a frozen LLM autonomously address, verify, execute, delete, and reinstall functional LoRA memories? |

## Fast package gate

```bash
dendritron smoke --json
pytest
```

## CPU research dependencies

```bash
pip install -e ".[research]"
python benchmarks/archive/dendritron_minsky_benchmark_v0_2.py
python benchmarks/archive/dendritron_optdigits_benchmark.py
```

Some archived scripts originally targeted `/mnt/data` or Colab `/content`. They are preserved as executed research artifacts. Change their output constants or run them in the documented environment before reproduction.

## Interpretation rules

- “Recorded” means the result file or executed notebook is included.
- “Tested” means the repository CI executes the mechanism on every supported Python version.
- A synthetic benchmark should not be described as a natural-data result.
- The Transformer result is a task-isolated functional-memory demonstration, not a claim of general lifelong learning.
- The Optical Digits model is a Dendritron-LVQ realization, not the whole canonical architecture.
- Hyperbolic geometry is selected by relational certificates; it is not asserted to dominate flat geometry on every classification task.

