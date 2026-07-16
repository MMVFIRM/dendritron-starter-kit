# Start Here

Choose the shortest path matching what you want to inspect.

| Goal | Start with | Then read |
| --- | --- | --- |
| Understand the primitive | `examples/quickstart_boolean.py` | `ARCHITECTURE.md`, `EQUATIONS.md` |
| Test continual local growth | `examples/continual_plasticity.py` | `EXTENDING.md` |
| Test Euclidean/hyperbolic compartments | `examples/mixed_geometry.py` | `ARCHITECTURE.md` |
| Build functional memory | `examples/functional_memory.py` | `TRANSFORMER_MEMORY.md` |
| Audit the research lineage | `benchmarks/archive/` | `EXPERIMENTS.md` |

The core install has only one runtime dependency: NumPy. Begin with `dendritron smoke`; it checks exact Boolean compilation, recursive parity, old-owner retention, functional memory routing, and basic hyperbolic numerics in seconds.

The archived scripts are scientific artifacts rather than package internals. They preserve the actual benchmark sequence and may use pandas, scikit-learn, PyTorch, Transformers, PEFT, or a GPU.

