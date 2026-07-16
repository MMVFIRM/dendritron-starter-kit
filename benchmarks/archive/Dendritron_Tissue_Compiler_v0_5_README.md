# Dendritron Tissue Compiler v0.5

## Purpose

v0.4 showed that immutable shared regions plus copy-on-write preserve ownership. v0.5 asks the harder question:

> Can the system discover reusable functional regions instead of receiving them from the engineer?

The compiler receives only complete eight-input Boolean task tables. It is not told the hidden shared input subsets, family assignments, or shared functions.

## Exact local decomposition

For a candidate local subset \(S\), the compiler asks whether a task can be written exactly as:

\[
F_t(x)=A(x_S)\oplus L_t(x_{\bar S}).
\]

The truth table is rearranged into a matrix whose rows are states of \(x_S\) and columns are states of the complementary variables. Such a decomposition exists exactly when:

\[
M_{r,c}=a_r\oplus \ell_c.
\]

The compiler canonicalizes the complement ambiguity by requiring:

\[
A(0,\ldots,0)=0.
\]

Tasks with the same canonical \(A\) are candidates for one shared Dendritron region. Candidate families are ranked by exact literal-cost savings.

## Benchmark construction

The raw task set contains:

- **16 tasks** generated from one hidden four-bit abstraction on bits \((0,2,5,7)\);
- **8 tasks** generated from another hidden abstraction on bits \((1,3,4,6)\);
- **4 balanced random functions** with no supported shared decomposition.

The compiler searches every input subset of arity one through four:

\[
\sum_{k=1}^{4}\binom{8}{k}=162
\]

candidate subsets.

## Discovery result

The compiler recovered:

| Family | Hidden subset | Recovered tasks | Exact recovery |
|---|---|---:|---:|
| A | \((0,2,5,7)\) | 16 | Yes |
| B | \((1,3,4,6)\) | 8 | Yes |

The four unsupported random tasks remained monolithic.

- Exact family grouping: **yes**
- Exact hidden subsets: **yes**
- Unsupported tasks forced into families: **zero**
- Reconstruction accuracy across all 28 tasks: **100%**

## Storage result

| Representation | Branches | Literal cost |
|---|---:|---:|
| Monolithic exact Dendritrons | 3,584 | 28,672 |
| Independently factored tasks | 898 | 5,636 |
| Automatically shared tissue | 722 | 4,932 |

The discovered tissue achieved:

\[
\boxed{5.81\times}
\]

literal-cost compression relative to the monolithic exact representation, while also saving 704 literal equivalents compared with duplicating the discovered abstraction for every task.

## Automatic specialization

Four members of Family A were then changed to use the same specialized shared function \(A'\). The compiler was rerun without being told which tasks had changed.

It recovered:

- original \(A\): **12 consumers**;
- specialized \(A'\): **4 consumers**;
- Family B: **8 consumers**;
- four random tasks: still monolithic.

Results:

- Original shared version preserved: **yes**
- Specialized shared version created: **yes**
- Accuracy of old consumers: **100%**
- Minimum accuracy across all tasks: **100%**
- Post-specialization literal compression: **5.78×**

The architecture discovered the functional fork from behavior rather than globally mutating the original factor.

## Architectural rule established

> Reusable regions can be discovered as exact shared functional factors, and unsupported tasks should remain separate rather than being forced into a common latent block.

This is the first automatic tissue compiler in the benchmark sequence. It transforms a set of monolithic functions into an explicit graph of:

- shared immutable Dendritron factors;
- private local factors;
- reusable composition operators;
- monolithic exceptions when factorization is unsupported.

## Limitations

- The compiler searches exact XOR decompositions only.
- The local arity is bounded at four.
- Complete truth tables are available.
- Search is combinatorial and intended only as a proof of architecture.
- Real data require approximate factor discovery, continuous representations, uncertainty, and partial certificates.
- The current compiler recompiles the small benchmark rather than maintaining a fully online dependency graph.

The next layer is an approximate continuous compiler that discovers coalitions under noise, creates branches endogenously, and verifies that shared factors remain sufficient for their consumers.
