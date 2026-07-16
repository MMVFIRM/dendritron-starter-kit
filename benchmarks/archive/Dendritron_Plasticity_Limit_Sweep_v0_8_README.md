# Dendritron Plasticity Limit Sweep v0.8

## Purpose

v0.7 showed that one Dendritron web can complete the full artificial-plasticity lifecycle. v0.8 asks where that behavior stops working.

The sweep varies:

- geometric separation between class regions;
- observation noise;
- ownership-certificate budget;
- developmental exposure;
- random world seed.

Forty-two complete developmental runs were performed.

## Two pass definitions

A structural event should not be treated as mandatory when the stream provides no reason for it. v0.8 therefore reports two different outcomes.

### Functional plasticity pass

The web must:

- finish near the true geometric oracle;
- preserve old modes;
- acquire new modes and classes;
- recover a dormant function;
- detect and repair controlled damage.

### Full-mechanism pass

In addition to the functional requirements, the run must visibly exercise the complete structural repertoire: growth, splitting, merging, and retirement.

This distinction prevents a nearly perfect run from being called a failure merely because no branch needed to split.

## Geometry boundary

The nominal separation-to-noise ratio is:

\[
\rho=\frac{\sqrt{2}\,s}{\sigma},
\]

where \(s\) is the class-center scale and \(\sigma\) is observation noise.

| Separation/noise | Oracle accuracy | Web accuracy | Functional pass rate | Full-mechanism pass rate |
|---:|---:|---:|---:|---:|
| 12.08 | 100.00% | 100.00% | 100% | 100% |
| 8.25 | 100.00% | 99.12% | 100% | 100% |
| 5.89 | 98.94% | 96.97% | 100% | 100% |
| 4.48 | 94.14% | 88.59% | 66.7% | 33.3% |
| 3.54 | 85.27% | 67.96% | 0% | 0% |
| 2.77 | 73.20% | 48.72% | 0% | 0% |

### Empirical boundary

- **Reliable full-plasticity boundary:** \(\rho\approx5.89\)
- **Functional majority boundary:** \(\rho\approx4.48\)
- **First complete functional failure:** \(\rho\approx3.54\)

At \(\rho\approx5.89\), the web stayed within 1.97 percentage points of the oracle while retaining old functions, acquiring new functions, reactivating dormant structure, repairing complete regional damage, and exercising the full structural repertoire across all seeds.

At \(\rho\approx4.48\), the oracle itself had fallen to 94.14%. The web averaged 88.59%, and only two of three seeds retained the full functional criteria. This is the transition zone.

Below \(\rho\approx3.54\), local ownership evidence became too ambiguous. Branch growth and specialization largely ceased, archives emptied, and classification fell far below the oracle. The current architecture cannot reliably discover ownership boundaries in that regime.

## Certificate budget

Moderate geometry was tested with 12, 24, 48, and 72 certificates per class.

| Certificates/class | Functional pass rate | Full-mechanism pass rate | Mean accuracy | Mean oracle gap |
|---:|---:|---:|---:|---:|
| 12 | 66.7% | 66.7% | 94.26% | 4.68 pp |
| 24 | 100% | 100% | 97.39% | 1.55 pp |
| 48 | 100% | 100% | 97.37% | 1.57 pp |
| 72 | 100% | 100% | 96.97% | 1.97 pp |

The smallest certificate budget that passed every seed was **24 per class**. More certificates did not monotonically improve accuracy because stricter ownership protection can reject otherwise useful branches. This is a genuine stability-plasticity tradeoff rather than a simple “more review is always better” relationship.

## Developmental exposure

Exposure results were non-monotonic across seeds. Full exposure passed every seed, while shorter schedules sometimes passed and sometimes missed recurrence or acquisition thresholds.

This means the current maintenance cadence is tied too closely to absolute step counts. A mature architecture should schedule splitting, consolidation, and retirement from event statistics rather than a fixed clock.

## What broke first

The failure was not arbitrary.

As overlap increased:

1. local support became less class-specific;
2. quarantine could no longer assign clean ownership;
3. split and merge evidence weakened;
4. archived structures became less distinguishable from active structures;
5. the router began losing more accuracy than the geometric oracle required.

The first major bottleneck is therefore:

> **Ownership discovery under overlapping manifolds.**

The current RBF geometry assumes that local distance is a useful proxy for functional ownership. At high overlap, it is not.

## Conclusion

The web is not infinitely overpowered. It has a measurable operating envelope.

Within that envelope, it exhibits artificial neural plasticity with explicit local ownership, structural development, dormancy, recurrence, and repair. Outside it, the current geometric router cannot infer sufficiently clean functional boundaries.

The next architecture problem is not whether plasticity exists. It is how to preserve plasticity when functions occupy deeply overlapping manifolds. That likely requires context-conditioned routing, learned local metrics, relational certificates, and higher-order coalitions rather than raw Euclidean proximity.
