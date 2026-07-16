# Contributing

Thank you for extending the Dendritron starter kit.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Pull requests

Keep the installable core dependency-light. Heavy research integrations should be optional extras or archived benchmarks. Every change to an architectural invariant needs a focused test and a short explanation in `docs/`.

For experimental results, include the seed, configuration, environment, hardware, executed output, and a clear distinction between observation and interpretation. Do not overwrite historical artifacts; add a new version.

## Architecture checklist

- Is every mutable branch or memory assigned to an owner?
- Does registration require a certificate?
- Can a failed candidate be quarantined?
- Does adding or specializing a function preserve old-owner signatures?
- Is geometry treated as routing support rather than ownership?
- Are local damage and repair visible in the event log?
- Are all numerical operations finite on degenerate inputs?

