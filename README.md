# Anonymous supplementary artifact

This repository contains the quantitative appendix and aggregate evidence tables
for a double-blind manuscript on functional emotion alignment. It is deliberately
limited to the material needed to inspect and reproduce the reported appendix.

## Contents

- `APPENDIX_COMPLETE.tex`: copy-ready replacement appendix.
- `RESULTS_MASTER.csv`: normalized ledger of 10,064 reported aggregate estimands.
- `build_appendix.py`: deterministic appendix/ledger builder using the 37 included
  aggregate CSV tables.
- `verify_release.py`: structural, provenance, reproducibility, and anonymity checks.
- `artifacts/`, `notebooks/`, and `outputs/`: saved aggregate tables at the paths
  expected by the builder.

Rebuild and validate with Python 3 (no third-party packages or model inference):

```bash
python3 build_appendix.py
python3 verify_release.py
```

The artifact excludes the original manuscript, author and affiliation metadata,
version-control history, raw human narratives, report- or participant-level rows,
model weights, caches, notebooks, and temporary files. The two originally
participant-indexed result tables were reduced to their published aggregate and
quartile summaries; identifier columns were removed.

For double-blind review, distribute this material only through the venue-approved
anonymous mirror. Do not give reviewers the identifiable repository-owner URL or
upload a local `.git` directory.
