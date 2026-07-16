# Dendritron–Minsky Benchmark v0.2

## Question

Can the original artificial-neuron primitive be replaced by a locally complete, recursively composable unit that learns Minsky-class Boolean functions while preserving functional ownership under growth and repair?

## Construction

For a local binary input \(x\in\{0,1\}^d\), define one branch for pattern \(p\):

\[
\chi_p(x)=\prod_{j=1}^{d}x_j^{p_j}(1-x_j)^{1-p_j}.
\]

On the Boolean cube, \(\chi_p(x)=1\) exactly when \(x=p\), and zero otherwise. Therefore any local Boolean function has the exact Dendritron representation

\[
f(x)=\sum_{p\in\{0,1\}^d} f(p)\chi_p(x).
\]

This gives a complete local function primitive. It has exponential worst-case branch cost in local dimension, so large functions must be built by composing bounded-input Dendritrons rather than swelling one unit indefinitely.

## Results

### 1. Exhaustive local universality

The benchmark tested all **65,536** possible four-input Boolean functions. The 16 local branch activations formed the identity basis over the 16 Boolean inputs, and every function was reconstructed exactly.

- Functions tested: **65,536**
- Exact reconstruction: **100%**
- Branch basis equals identity: **true**

### 2. Learned XOR, recursively scaled parity

One two-input XOR Dendritron was compiled from its four-row truth table. The same intact local function was cloned and recursively composed into balanced parity tissues.

- Maximum input size: **4,096 bits**
- Minimum accuracy across all tested sizes: **100%**
- Dendritrons at 4,096 bits: **4,095**
- Branches: **8,190**
- Balanced depth: **12**

A direct one-Dendritron minterm representation of 4,096-bit parity would require \(2^{4095}\) positive branches. Recursive composition changes the cost to \(n-1\) two-input Dendritrons.

### 3. Learned connectedness rule

A six-input local reachability function—one active-state bit plus the cell and four neighbors’ reachability states—was compiled from all 64 local configurations. The identical learned cell was then deployed across recurrent two-dimensional tissues.

| Grid | Local Dendritrons | Accuracy vs BFS |
|---|---:|---:|
| 8×8 | 64 | 100% |
| 16×16 | 256 | 100% |
| 32×32 | 1,024 | 100% |

The local function was learned once. Grid scale changed only the number and organization of intact cells.

### 4. Sequential ownership benchmark

Twelve balanced random four-input Boolean tasks were presented sequentially.

| Model | Stored parameters / equivalents | Acquisition | Final accuracy | Mean forgetting |
|---|---:|---:|---:|---:|
| Owned Dendritron Web | 504 | 100% | 100% | 0% |
| Shared Context MLP | 577 | 100% | 56.77% | 47.16% |
| Independent MLP Bank | 1,164 | 100% | 100% | 0% |

The shared MLP was capable of learning every task at acquisition time. Its failure was preservation: later updates altered the shared hidden substrate supporting earlier tasks.

The independent MLP bank confirms the causal point. Local ownership can prevent forgetting even with conventional units, but it requires a complete separate MLP per task. The Dendritron web retained all tasks with fewer stored scalar equivalents and branch-level repairability.

### 5. Non-destructive growth

Adding each new function region changed **zero** signatures belonging to earlier regions.

This establishes the tested invariant:

\[
\Delta\Theta_{R_j}=0\quad \forall j<t
\]

when task \(t\) is allocated a new owned region.

### 6. Local damage and repair

One branch was removed from task region 5.

- Accuracy before damage: **100%**
- Accuracy after branch damage: **93.75%**
- Failed region detected: **region 5 only**
- Regions structurally changed by damage: **region 5 only**
- Accuracy after certificate reconstruction: **100%**
- Non-target regions changed by repair: **none**

## What v0.2 establishes

1. A Dendritron can be a complete local Boolean-function primitive rather than a scalar affine threshold unit.
2. The primitive can be learned locally from a finite certificate.
3. Bounded-input Dendritrons compose into parity and connectedness systems without changing the primitive.
4. Functional ownership eliminates silent parameter interference in the tested sequential setting.
5. Local verification and repair restore damaged capability without global retraining.

## What it does not establish

- The local truth-table compiler currently requires complete Boolean certificates.
- Worst-case single-unit branch count remains exponential in local input dimension.
- The sequential test uses an explicit task router rather than a learned semantic router.
- The shared MLP control uses no replay, EWC, parameter isolation, or frozen task modules.
- Continuous representation learning, natural data, differentiable branch creation, and energy efficiency remain untested.
- This is not yet a lower-bound proof showing that all equivalent MLP constructions require more compute or parameters.

## Next benchmark

The next architecture layer should replace the explicit task ID with a learned ownership router while preserving the no-silent-interference invariant. It should add:

1. continuous vector inputs;
2. branch growth and splitting from incomplete samples;
3. shared immutable abstractions plus copy-on-write specialization;
4. semantic routing without task labels;
5. matched-compute comparisons against replay, EWC, MoE, and progressive networks.
