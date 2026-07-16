# Extending the Kit

## Add a new branch type

Implement three operations:

1. `activation(values)` or `score(values)`;
2. a local update that cannot touch another owner;
3. serializable state sufficient for an integrity signature.

Add tests for dimensions, finite numerics, ownership mismatch, damage isolation, and repair.

## Add a geometry

Add projection and distance functions beside `geometry.py`, then expose the geometry through a `Chart`. A geometry admission policy must be evaluated against both the candidate owner's evidence and protected certificates from other owners. Do not change ownership simply because a chart changes.

## Add a memory implementation

Create a `MemoryPack` with:

- a callable functional module;
- a generative address model over a shared coordinate;
- an optional independent verifier;
- an unchanged validation gate;
- metadata and reproducible hashes.

Deletion must remove the pack from candidate generation. Reinstallation must verify the stored asset before registration.

## Research contribution checklist

- State the certificate and registration threshold.
- Separate executed results from planned experiments.
- Include at least one interference or retention test.
- Include a damage and repair test when mutable local state is introduced.
- Include a flat baseline when proposing hyperbolic geometry.
- Include an ablation when introducing routing or verification machinery.
- Record seeds, package versions, hardware, and output hashes.

