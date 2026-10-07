# MIMO neural experiments for intraday PV residual trajectories

This document compares two experimental MPC forecasters against the frozen
PV day-ahead model:

- a feed-forward MIMO MLP;
- an encoder–decoder MIMO LSTM.

Neither replaces the pointwise CatBoost residual model yet. The purpose is to
test whether joint rest-of-day residual trajectories add usable information to
MPC.

The reusable implementation is in `src/energy/modeling/pv_mimo.py`. Runnable
MLP and LSTM cells are at the end of `working/data_analysis.ipynb`.

## Forecast definition

The day-ahead model produces a frozen forecast before delivery:

```text
pv_hat_da[t]
```

At decision hour `tau`, either MIMO model produces a residual trajectory for
future solar-active hours:

```text
(residual_hat[tau, 1], ..., residual_hat[tau, H_max])
residual[t] = pv_actual[t] - pv_hat_da[t]
pv_hat_mpc[tau, t] = max(pv_hat_da[t] + residual_hat[tau, lead(tau, t)], 0)
```

`H_max` is fixed to the maximum observed future solar-active horizon. For the
current 2024 profile and clear-sky activity rule it is 12 hours. Shorter
end-of-day paths are zero-padded; an explicit target mask excludes padding
from loss and metrics.

## Inputs and availability

Each sample is one `(delivery date, decision hour)` pair. Both models use:

- the latest `K` completed active-hour values of actual PV, frozen day-ahead
  PV, and their residuals;
- the full remaining trajectory of frozen day-ahead PV, forecast weather,
  calendar values, and solar geometry;
- no actual PV or residual at the current or a future hour.

The historical day-ahead prediction is cross-fitted in outer training and then
frozen before residual samples are built. Complete local-calendar weeks are
kept together in every split.

## Architectures

### MIMO MLP

The MLP flattens the history and known future trajectories, standardises the
resulting vector, and predicts all 12 residuals simultaneously. Its current
configuration has two hidden layers of 48 units, dropout, AdamW and masked
Huber loss.

### Encoder–decoder MIMO LSTM

The LSTM preserves the sequence structure explicitly:

1. An encoder reads the completed actual-PV, day-ahead-PV and residual sequence
   in chronological order.
2. A decoder is initialised from that state and reads the known future sequence
   of day-ahead PV, weather, calendar and solar features.
3. A shared linear head maps every decoder state to one residual value.

There is no teacher forcing: the decoder never receives future factual PV or
future residuals. Variable history and future lengths use packed sequences;
padding cannot affect the LSTM state or the loss. The current compact model
has one 24-unit LSTM layer and is deliberately smaller than the MLP because
there are only 344 independent delivery days.

## Validation protocol

Three season-balanced outer folds of complete local-calendar weeks evaluate the
entire two-stage process. For each outer fold:

1. create out-of-fold day-ahead predictions inside outer training and frozen
   outer-test day-ahead predictions;
2. build MIMO decision-time samples separately for training and test weeks;
3. use one inner whole-week fold to select the early-stopping epoch;
4. refit on all outer-training weeks for exactly that epoch and score the outer
   test trajectories.

Install the experiment dependencies and run the direct-residual setup cells,
then the MLP and LSTM cells in the notebook:

```bash
python -m pip install -e '.[train,neural]'
```

## Initial 2024 comparison

All variants below use the same future solar-active `(decision hour, target
hour)` pairs. Values are the mean and sample standard deviation across the
three outer folds.

| Variant | MAE, W | MAE fold SD, W | RMSE, W | RMSE fold SD, W |
| --- | ---: | ---: | ---: | ---: |
| Frozen day-ahead baseline | 18.07 | 1.08 | 34.05 | 2.21 |
| MIMO MLP residual correction | 17.46 | 0.59 | 33.12 | 1.22 |
| Encoder–decoder MIMO LSTM correction | 17.66 | 1.12 | 33.41 | 2.49 |

The MLP gains 0.61 W MAE against day-ahead; the LSTM gains 0.42 W. Both are
most useful at lead 1, while late-hour corrections are mixed. The MLP is the
best neural result, but its advantage over the direct CatBoost residual model
is too small relative to the three-fold variation to claim a reliable winner.

For the current single-site dataset, retain the direct CatBoost residual and
the MIMO MLP as the two candidate models. The LSTM is a useful completed
comparison: it proves the sequence-to-sequence formulation works under the
same leakage-safe protocol, but it does not justify greater deployment
complexity. More sites or multiple years of data would make a stronger case
for reconsidering recurrent or attention-based architectures.
