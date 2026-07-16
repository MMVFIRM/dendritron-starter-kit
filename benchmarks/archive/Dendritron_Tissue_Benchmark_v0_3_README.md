# Dendritron Tissue Benchmark v0.3

## Purpose

v0.2 established exact local Boolean function compilation, recursive parity, learned connectedness, explicit functional ownership, and local repair. v0.3 removes the explicit task identifier and asks whether continuously learned Dendritron regions can enter a shared semantic web without silently stealing old functions.

## Continuous Dendritron region

Each region owns a small set of locally learned radial branches:

\[
a_b(x)=\exp\!\left(-\frac{\|x-\mu_b\|^2}{2s^2}\right).
\]

The current optimized router uses the equivalent monotone support score

\[
S_R(x)=-\frac{\min_b\|x-\mu_b\|}{s_R}.
\]

The selected label is the active region with maximum local support. No task ID is supplied.

## Quarantine activation

A candidate region is trained outside the active web. Before activation, the router evaluates all protected certificates belonging to existing regions:

\[
\hat y_{\text{before}}(x)=\hat y_{\text{after}}(x)
\qquad \forall x\in\mathcal C_{\text{old}}.
\]

The candidate is admitted only if it changes zero protected old predictions. This does not prove global noninterference, but it makes interference explicit, testable, and rejectable rather than silent.

## Results

### Continuous class-incremental growth

Ten synthetic eight-dimensional classes were learned sequentially. Each class was a two-mode continuous distribution represented by two locally owned RBF branches.

- Task ID supplied to router: **no**
- Base regions accepted: **10 of 10**
- Protected old decisions changed during growth: **0**
- Final mean accuracy: **100%**
- Minimum class accuracy: **100%**
- Inference scalar equivalents after the additional resolved region: **187**
- Protected certificate scalars: **5,632**

### Shared MLP controls

| Model | Parameters | Replay memory scalars | Acquisition | Final accuracy | Mean forgetting |
|---|---:|---:|---:|---:|---:|
| Dendritron semantic web | 187 inference equivalents | 5,632 audit certificates | 100% | 100% | 0% on tested sequence |
| Shared MLP, no replay | 1,226 | 0 | 100% | 60.0% | 44.44% |
| Shared MLP, replay 20/class | 1,226 | 1,800 | 100% | 99.45% | 0.61% |

Replay nearly repaired the MLP control. This is important: old data can compensate for shared-substrate interference by repeatedly re-optimizing the block. The architectural distinction is that the Dendritron web preserved old regions without modifying or retraining them.

### Ownership-conflict test

A proposed new region was created as a relabeling of class 0 and given stronger support over the same manifold.

- Candidate accepted: **no**
- Protected old predictions it would have changed: **65**

The system refused to invent a boundary unsupported by the input evidence.

A second candidate retained similarity to class 0 but added a genuinely distinguishing eighth-dimensional feature.

- Resolved candidate accepted: **yes**
- New-class accuracy: **100%**
- Accuracy on the ten old classes after activation: **100%**

This establishes a crucial rule:

> When two claimed functions are not separable in the observed representation, the architecture should detect the ownership conflict and refuse activation rather than silently deform the old space.

### Local damage and repair

Region 4 was disabled.

- Class accuracy before damage: **100%**
- Class accuracy after damage: **0%**
- Failed region identified: **4 only**
- Structurally changed region: **4 only**
- Accuracy after local certificate reconstruction: **100%**
- Non-target regions changed during repair: **none**

## What v0.3 adds

1. Continuous inputs rather than complete Boolean truth tables.
2. Branch growth from finite samples through local clustering.
3. Semantic routing without task labels.
4. Quarantine and certificate-based admission of new regions.
5. Explicit rejection of a mathematically unsupported ownership split.
6. Local reconstruction without retraining the active web.

## What remains unsolved

- Certificates protect sampled old behavior, not every possible point on an old manifold.
- The router is still a simple local-support competition rather than a recursively learned Dendritron routing tissue.
- The synthetic classes are deliberately separable.
- Shared abstractions and copy-on-write specialization are not yet implemented.
- Region splitting, merging, redundancy, and repair under partial rather than total damage need stronger tests.
- Real image, language, and temporal data remain untested.

## Next architecture layer

The next build should make routing itself a Dendritron tissue. It should learn reusable immutable feature regions, form functions as coalitions of those regions, and fork shared functions through copy-on-write when new evidence requires specialization.
