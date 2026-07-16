# Dendritron Tissue Benchmark v0.4

## Purpose

v0.4 tests **shared abstraction without shared mutability**. The computational primitive is held constant while three ownership policies are compared:

1. **Shared mutable:** every task references one abstraction and specialization mutates it globally.
2. **Independent copies:** every task stores a complete private abstraction.
3. **Versioned copy-on-write:** tasks reuse an immutable abstraction until specialization creates or reuses a content-addressed fork.

This isolates ownership semantics from representational power.

## Local Dendritron

Each exact Boolean Dendritron represents a local function through positive minterm branches:

\[
\chi_p(x)=\prod_j x_j^{p_j}(1-x_j)^{1-p_j},
\]

\[
f(x)=\sum_{p:f(p)=1}\chi_p(x).
\]

The local four-input domain is finite, so the certificate is exhaustive rather than sampled.

## Composite task

Thirty-two tasks share a four-bit abstraction \(A\), own separate four-bit local functions \(L_t\), and reuse one immutable XOR combiner:

\[
F_t(s,\ell)=A(s)\oplus L_t(\ell).
\]

A specialization changes six of the sixteen states of \(A\) for one task.

## Single-specialization result

| Architecture | Specialized task | Mean sibling accuracy | Existing nodes changed |
|---|---:|---:|---:|
| Shared mutable | 100% | 62.5% | 1 shared parent |
| Independent copies | 100% | 100% | 0 unrelated nodes |
| Versioned copy-on-write | 100% | 100% | 0 existing nodes |

The globally mutable architecture learns the requested specialization but silently changes the function used by all siblings. Copy-on-write creates a new intact Dendritron and rebinds only the requesting task.

## Content-addressed specialization families

Sixteen tasks request four repeated specialization variants.

- Specialization requests: **16**
- Unique forks created: **4**
- Existing forks reused: **12**
- Active shared versions including the parent: **5**
- Mean and minimum task accuracy: **100%**

This demonstrates that copy-on-write does not require one private copy per consumer. Identical functional variants converge on the same immutable region.

## Active storage after specialization

| Architecture | Active shared versions | Branches | Literal cost |
|---|---:|---:|---:|
| Shared mutable | 1 | 266 | 1,060 |
| Independent copies | 32 | 514 | 2,052 |
| Versioned copy-on-write | 5 | 298 | 1,188 |

Copy-on-write remains close to fully shared storage while preserving exact sibling behavior.

## Damage and rollback

One truth-table state in the shared parent was corrupted after half the tasks had forked away from it.

- Dependents of damaged parent: **16**
- Nondependents: **16**
- Failed node identified: **exactly one**
- Mean dependent accuracy after damage: **93.75%**
- Mean nondependent accuracy after damage: **100%**
- Nondependent tasks changed: **0**
- Accuracy after local rollback: **100%**
- Other nodes modified during repair: **0**

The dependency graph identifies which functions are exposed to damage, and an immutable snapshot repairs the failed region without global retraining.

## Architectural rule established

> Shared use must not imply shared write permission.

A Dendritron region may have many consumers. A consumer that requires specialization either reuses an existing compatible version or creates a new version. The parent remains intact.

## Limitations

- Functions are exact finite Boolean tables.
- The shared abstraction is specified rather than discovered in v0.4.
- Content hashes and full snapshots are practical here because the local domains are tiny.
- Real continuous representations require compressed certificates, probabilistic verification, and learned factorization.
- Version garbage collection, merging, and long lineage management remain open.

v0.5 addresses the largest immediate limitation by discovering reusable abstractions automatically from task behavior.
