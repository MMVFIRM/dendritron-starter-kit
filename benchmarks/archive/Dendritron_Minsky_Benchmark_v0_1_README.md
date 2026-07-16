# Dendritron–Minsky Benchmark v0.1

## What this build establishes

This benchmark is a constructive witness, not yet a general superiority claim.

1. **Primitive expressivity:** a single two-branch Dendritron computes XOR exactly, while one affine threshold perceptron cannot.
2. **Compositional scaling:** a balanced tissue computes `n`-bit parity with `n-1` intact Dendritrons, `2(n-1)` active branches, and depth `ceil(log2 n)`.
3. **Recursive closure:** two complete 8-bit parity regions plus one ordinary XOR Dendritron form a 16-bit higher-order Dendritron with the same external input/output contract.
4. **Non-destructive growth:** attaching dormant capacity changes no existing output.
5. **Local ownership and repair:** damaging one node changes only that node, its local verifier identifies it, and replacing only that node restores the global function.
6. **Connectedness:** a 2-D tissue of local reachability Dendritrons matches an exact BFS oracle on balanced connected/disconnected test sets.

## Current run

- Single-Dendritron XOR: **100%**
- Recursive parity: **100%** from 2 through 1,024 input bits on 5,000 random examples per size
- Recursive closure, 16-bit parity: **100%**
- Dormant growth maximum output change: **0**
- 64-bit parity before damage: **100%**
- After disabling one locally owned node: **49.71%**
- Local failed node detected: **exactly `(layer 0, node 3)`**
- After local reconstruction: **100%**
- Non-target nodes modified during repair: **none**
- Connectedness vs BFS: **100%** on 1,000 8×8 and 500 16×16 examples

## What this does not establish yet

- superiority over optimized MLPs on continuous real-world data;
- learnable sparse routing at million-unit scale;
- energy efficiency on specialized hardware;
- sample-efficiency advantages under matched parameter and compute budgets;
- a formal lower bound separating Dendritron Tissue from all MLP constructions.

Those are the next benchmark layers.
