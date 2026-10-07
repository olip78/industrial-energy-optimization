# Parallel multi-site PV experiment

This experiment is separate from the original single-site `1a` workflow. It
asks a narrow question: does pooling a few related PV profiles help the
pointwise day-ahead and intraday residual forecasts used by MPC?

## Isolation contract

No existing single-site input, model or result is replaced.

| Purpose | Parallel location |
| --- | --- |
| Multi-site day-ahead training data | `data/features/pv_multisite_v1/` |
| Dataset manifest and site metadata | `data/metadata/pv_multisite_v1_*` |
| Backtest artefacts | `artifacts/experiments/pv_multisite_v1/2024/` |
| Training implementation | `src/energy/training/pv_multisite_experiment.py` |

The complete parallel branch can therefore be inspected or removed later
without changing the baseline `1a` tables under `data/features/day_ahead_pv/`.

## Sites and metadata

The MPVBench source supplies five measured PV profiles. `pv_multisite_v1`
contains only `1a`, `2a` and `2b`:

| Profile | Rows in 2024 table | Reason |
| --- | ---: | --- |
| `1a` | 8,277 | Existing working profile |
| `2a` | 8,277 | Usable PV profile |
| `2b` | 8,277 | Usable PV profile |
| `1b` | — | Excluded: 147 all-zero days in the available source period |
| `1c` | — | Excluded: 313 all-zero days in the available source period |

Every row has `site_id` plus `site_id_1a`, `site_id_2a` and `site_id_2b`.
These are numeric one-hot features. With only three known installations, this
is simpler and more transparent than an embedding.

The source publishes a detailed installation description only for `1a`. It
does not give confirmed panel coordinates for `2a` or `2b`. All profiles
therefore use the existing Pforzheim weather and solar-geometry proxy
(48.89, 8.70) and the data records that provenance. Latitude and longitude are
not model features because they would have the same imputed value in every
row. Observed peaks are descriptive metadata, not assumed installed power.

## Leakage controls

The multi-site data retains the existing provisional `Europe/Berlin` PV time
convention and fixed-lead historical weather contract. The outer split assigns
calendar weeks, rather than individual rows, to train or test. Crucially, a
test week is held out for **all** three profiles: actual PV from `2a` or `2b`
cannot reveal weather errors for `1a` in the same test period.

Day-ahead predictions inside outer training are generated out of fold before
residual targets and factual residual lags are constructed. MIMO trajectories
are grouped by `(site_id, delivery_date_local)`, so the historical sequence
never crosses installation boundaries.

## Models compared

The experiment fits one pooled model for each approach. Site one-hot features
are available to the day-ahead CatBoost model and throughout the MIMO future
trajectory.

1. frozen pooled day-ahead CatBoost baseline;
2. pointwise direct CatBoost residual correction;
3. MIMO MLP residual trajectory correction;
4. encoder–decoder MIMO LSTM residual trajectory correction.

The direct and MIMO correction models receive actual PV only from completed
hours of the **same** site. Each model is reported both pooled and separately
for `1a`, `2a` and `2b`.

## Reproduce

```bash
python -m pip install -e '.[train,neural]'
energy build-multisite-pv-data --project-root . --year 2024
energy run-pv-multisite-experiment --project-root . --year 2024
```

The second command writes fold-level scores, lead-time scores, epoch-selection
records and a JSON summary to the dedicated artefact directory. Results will
be added here after the first complete three-fold run.

## First complete 2024 result

The full three-fold run completed with four model variants. The scores below
are the mean MAE and RMSE in watts over the same future solar-active
`(decision hour, target hour)` pairs; standard deviations are across folds.

### Pooled rows

| Variant | MAE, W | MAE fold SD, W | RMSE, W | RMSE fold SD, W |
| --- | ---: | ---: | ---: | ---: |
| Frozen day-ahead CatBoost | 34.68 | 1.17 | 56.73 | 2.65 |
| Direct residual CatBoost | 34.50 | 1.47 | 57.94 | 3.70 |
| MIMO MLP residual | **33.81** | 1.58 | **56.30** | 3.53 |
| Encoder–decoder MIMO LSTM residual | 33.95 | 1.47 | 56.47 | 2.87 |

### MAE by profile

| Profile | Day-ahead | Direct CatBoost | MIMO MLP | MIMO LSTM |
| --- | ---: | ---: | ---: | ---: |
| `1a` | 19.56 | **18.91** | 19.07 | 19.28 |
| `2a` | 57.01 | 57.59 | **55.39** | 55.55 |
| `2b` | 27.47 | 27.01 | **26.98** | 27.03 |

The MLP has the best pooled result and gives the largest gain at short leads.
However, the multi-site model does **not** improve the original target profile
`1a`: its best result is the direct CatBoost correction, and all pooled
variants are weaker than the earlier single-site `1a` backtest.

This is not evidence that additional PV data is intrinsically unhelpful. It is
evidence that these particular profiles are not exchangeable after a site
one-hot feature alone. Their panel orientation, tilt, shading, exact location
and nominal capacity are mostly unknown; all three profiles also share the
same imputed weather/solar coordinate. Pooled training therefore adds examples
with different, partially unobserved physical response functions.

The sensible next experiment is not to deploy the pooled model. It is to add
only defensible site metadata where available, use per-site target scaling
fitted inside each outer fold, and evaluate a shared model with a site-specific
output component. The present result is retained as the honest one-hot
baseline for that comparison.
