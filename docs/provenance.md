# Research provenance

SAGE-AVO is organized as a versioned research workflow. Each generated artifact
records its configuration, source inputs, array shape and dtype, checksum, and
Git commit when available. Notebook 01 exports the canonical `s01data/v003`
structure and elastic-background contract used by downstream stages.

The repository separates three classes of material:

- reusable algorithms and data contracts in `src/sage_avo`;
- concise scientific workflows in `notebooks` and `scripts`;
- machine-local field inputs and generated artifacts under ignored paths.

Field SEG-Y, horizon interpretations, LAS logs, trained checkpoints, and derived
field arrays remain under their governing data permissions. Public commits
contain configuration templates, algorithms, tests, documentation, and figures
that are authorized for distribution.

Versioned manifests, input checksums, realization-level split identifiers, and
explicit horizon roles preserve computational provenance. CI runs
`scripts/check_public_repo.py` to detect personal paths, common secret formats,
and unexpectedly large files.
