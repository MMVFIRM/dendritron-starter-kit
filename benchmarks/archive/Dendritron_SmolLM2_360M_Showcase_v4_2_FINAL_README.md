# Dendritron SmolLM2-360M v0.4.2

v0.4.2 is a registry-repair gate.

It can import and SHA-256 verify the four registered v0.4.1 adapters from:

`/content/dendritron_smollm2_360m_v4_1`

It writes the repaired runtime to:

`/content/dendritron_smollm2_360m_v4_2`

The quarantined character-order memory is replaced by `endpoint_match`, which
asks whether the first and final color tokens in a sequence are identical.

When the prior directory is unavailable, the notebook retrains all five packs
and remains self-contained.
