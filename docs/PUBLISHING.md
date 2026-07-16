# Publish to GitHub

The package is repository-ready but does not guess your GitHub account or organization.

1. Create an empty repository named `dendritron-starter-kit`.
2. Replace `OWNER` in the three `[project.urls]` entries in `pyproject.toml`.
3. From this directory, run:

```bash
git init -b main
git add .
git commit -m "Initial Dendritron complete starter kit"
git remote add origin git@github.com:OWNER/dendritron-starter-kit.git
git push -u origin main
```

4. Enable GitHub Discussions, Issues, and private vulnerability reporting if desired.
5. Add repository topics: `dendritron`, `continual-learning`, `hyperbolic-geometry`, `functional-memory`, and `neural-architecture`.
6. Create a `v0.1.0` release after CI passes and attach the wheel, source distribution, and repository archive.

Do not commit model checkpoints, adapters, tokens, `.env` files, or proprietary datasets. The `.gitignore` already excludes the common forms.

