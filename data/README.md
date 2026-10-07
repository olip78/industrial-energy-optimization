# Local data layout

Only provenance manifests and source licence/metadata files are committed to
Git. Downloaded observations, materialized joins and model features remain
local because they are generated artifacts and make the repository difficult
to review.

```text
raw/        original source responses
processed/  cleaned source-specific tables
curated/    canonical hourly tables and DuckDB materializations
features/   model-specific point-in-time feature views
metadata/   versioned coverage, quality and lineage manifests
```

Build the annual feature datasets after collecting the required sources:

```bash
energy build-training-data --project-root . --year 2024
```

The feature builders enforce the source-availability boundary documented in
[`docs/training_datasets_2024.md`](../docs/training_datasets_2024.md). Do not
commit downloaded or derived tables. If the project later needs shared binary
data, use an object store or DVC-style remote with checksums rather than Git.
