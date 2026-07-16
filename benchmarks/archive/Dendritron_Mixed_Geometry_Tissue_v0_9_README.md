# Dendritron Mixed-Geometry Tissue v0.9

## Question

Can a plastic Dendritron web compartmentalize Euclidean and hyperbolic spaces, select the useful local geometry, move functional evidence among charts without changing ownership, and retain damage repair and continual-growth behavior?

## Architecture

Each branch retains one functional owner but exposes a local chart bank:

\[
\mathcal C_b=\{C_{b1},\ldots,C_{bK}\},
\qquad
C_{bk}=(P_{bk},\mathcal M_{bk},c_{bk},\sigma_{bk}).
\]

A chart contains a projection or compartment \(P_{bk}\), a geometry \(\mathcal M_{bk}\), optional curvature \(c_{bk}\), and a local support radius \(\sigma_{bk}\).

The branch support is:

\[
s_b(x)=\sum_k \pi_{bk}
\exp\left[-\frac{d_{bk}(P_{bk}x,P_{bk}\mu_b)^2}{2\sigma_{bk}^2}\right].
\]

The owner of the branch does not change when chart weights \(\pi_{bk}\) change. Geometry is therefore a routing property of an owned function, not the identity of the function itself.

### Admission before activation

A new branch is first evaluated independently through every chart. Its admission chart is the chart producing the least violation of protected old ownership certificates:

\[
k^*=\arg\min_k\max_{x\in\mathcal C_{\text{old}}}
\left[s_{b,k}(x)-s_{\operatorname{owner}(x)}(x)-m\right]_+.
\]

The branch enters quarantine through \(k^*\). Later local evidence may spread its chart mixture or move it to another chart.

This rule fixed an observed failure in the first hybrid run: equal chart weights allowed an irrelevant Euclidean chart to reject otherwise safe hyperbolic regions.

## Tests

All reported classification results use three seeds: 11, 29, and 47. The streaming protocol includes new modes, new classes, specialization, inactivity, recurrence exposure, complete regional damage, local damage detection, and repair.

### 1. Flat isotropic control

At the moderate v0.8 overlap point, no geometry should possess a structural advantage.

| Router | Mean final accuracy |
|---|---:|
| Full-space Euclidean | 0.9669 |
| Full-space hyperbolic | 0.9628 |
| Full-space adaptive mixture | 0.9568 |

Hyperbolic geometry did not magically solve unstructured Gaussian overlap. This is a necessary negative control.

### 2. Hierarchical compartment

The observations contained a two-dimensional hierarchical compartment plus ten noisy or weak dimensions.

| Router | Mean final accuracy | Pass rate |
|---|---:|---:|
| Full-space Euclidean | 0.5274 | 0% |
| Hyperbolic tree compartment | 0.9788 | 100% |
| Compartmentalized Euclidean bank | 0.9965 | 100% |
| Mixed Euclidean/hyperbolic bank | **0.9975** | 100% |

The largest gain came from isolating the relevant compartment. Hyperbolic routing passed, but this leaf-classification objective did not require negative curvature once the correct two-dimensional compartment was available.

### 3. Hybrid stream

One tissue simultaneously received:

- flat functions in dimensions 0–5;
- hierarchical functions in dimensions 6–7;
- nuisance dimensions 8–11;
- a class whose old modes occupied the flat compartment and whose novel mode occupied the hierarchical compartment.

| Router | Mean final accuracy | Old functions | New functions | Pass rate |
|---|---:|---:|---:|---:|
| Full-space Euclidean | 0.8099 | 0.8673 | 0.6951 | 0% |
| Full-space hyperbolic | 0.7969 | 0.8627 | 0.6654 | 0% |
| Compartmentalized Euclidean bank | 0.9677 | 0.9642 | 0.9747 | 100% |
| Symmetric mixed bank | **0.9815** | **0.9796** | **0.9852** | 100% |

The mixed bank contained:

- Euclidean flat chart;
- Euclidean tree-compartment chart;
- hyperbolic tree-compartment chart;
- Euclidean full-space fallback.

The tissue performed an average of 7.67 local chart switches per run while keeping functional ownership explicit. Complete regional damage was detected and repaired locally in an average of 2.33 samples.

### 4. Hierarchical-distance limit test

Classification does not necessarily require hyperbolic distance. A direct relational test therefore measured how well two-dimensional Euclidean and hyperbolic charts preserve graph distance in balanced binary trees.

At depth 10, containing 2,047 nodes and 1,024 leaves:

| Geometry | Normalized distance RMSE | Correlation with graph distance |
|---|---:|---:|
| Euclidean | 0.3601 | 0.6146 |
| Hyperbolic | **0.0994** | **0.8901** |

As tree depth increased from 3 to 10:

- Euclidean distortion worsened;
- hyperbolic distortion steadily decreased after scale fitting;
- hyperbolic correlation with graph distance remained near 0.89.

A relational certificate minimizing graph-distance stress therefore selects hyperbolic geometry at every tested depth.

## Result

The v0.9 web demonstrates:

1. local geometry banks attached to persistent functional owners;
2. compartment-specific Euclidean and hyperbolic charts;
3. safe chart selection before branch admission;
4. local chart-weight adaptation and switching;
5. successful operation in mixed-geometry streams;
6. local damage detection and repair;
7. geometry selection based on the owned function's certificate type.

## Correct interpretation

The result is not:

> Hyperbolic geometry should replace Euclidean geometry everywhere.

It is:

> A Dendritron region should use the geometry that preserves the relations required by its owned function.

For local isotropic discrimination, Euclidean geometry may be sufficient. For hierarchy, ancestry, and tree-distance preservation, hyperbolic geometry becomes increasingly advantageous. A single web can contain both and move evidence among them without surrendering functional ownership.

## Remaining limitations

- Chart projections are supplied by a bounded chart library rather than discovered from arbitrary feature subsets.
- Curvature is fixed within each hyperbolic chart.
- The chart verifier currently uses classification margins or relational stress; a unified verifier has not been learned end to end.
- The mixed benchmark is synthetic and low dimensional.
- Chart migration is local, but chart creation and curvature splitting are not yet endogenous.

The next version should learn projection, curvature, and certificate type jointly, while preserving the admission and ownership invariants established here.
