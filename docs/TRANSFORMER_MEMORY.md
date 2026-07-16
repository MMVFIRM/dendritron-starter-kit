# Transformer Memory Packs

## Reference mechanism

The v0.4.2 showcase uses a frozen `HuggingFaceTB/SmolLM2-360M` backbone and five independent LoRA memories:

- `sum_threshold`;
- `vowel_majority`;
- `balanced_brackets`;
- `alternating_sequence`;
- `endpoint_match`.

Each pack earns registration at an unchanged 80% validation gate. At inference time, the prompt does not include a task identity. A pooled frozen hidden state becomes the shared coordinate. Address heads select candidates, PPCA verifiers bind a candidate, and only the selected adapter executes.

## Recorded quick-mode configuration

| Setting | Value |
| --- | ---: |
| Backbone | SmolLM2-360M |
| Train examples per memory | 320 |
| Validation examples per memory | 100 |
| Test examples per memory | 120 |
| LoRA rank | 8 |
| Shared coordinate | 32 |
| Verifier rank | 6 |
| Raw examples retained by packs | 0 |

## Run

Use an A100-class Colab runtime for the full formation benchmark:

```bash
pip install -e ".[transformer]"
python benchmarks/archive/dendritron_smollm2_360m_showcase_v4_2_FINAL.py --help
```

Read the archived v0.4.2 README for the exact previous-checkpoint import path and output path. The script can retrain all five packs when the v0.4.1 directory is unavailable.

## Integrity gates

The experiment checks:

- only adapter parameters are trainable;
- non-adapter sentinels remain unchanged during each formation run;
- the canonical frozen-backbone SHA-256 remains unchanged;
- old memory predictions survive new registrations;
- checkpoint reload preserves predictions and candidate sets;
- uninstall excludes the deleted adapter;
- reinstall restores adapter bytes and predictions.

Raw-logit equality is stricter than functional equivalence and was the only exact gate that did not pass in the recorded v0.4.2 run. Prediction, candidate, hash, deletion, and reinstall gates passed.

