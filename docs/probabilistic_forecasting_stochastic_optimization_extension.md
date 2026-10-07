# Probabilistic Forecasting and Stochastic Optimization Extension

## Purpose

This document describes a proposed probabilistic extension of the
existing deterministic energy-optimization design. It is intended as
input for updating the main design document.

The central architecture is deliberately simple:

**forecast -\> generate joint scenarios -\> optimize across scenarios
-\> execute**

The same machinery is used twice:

1.  **Day Ahead:** generate joint PV/price scenarios using all
    information available before the Day-Ahead decision and optimize the
    Day-Ahead schedule.
2.  **Intraday / MPC:** as new observations become available, update
    forecasts, generate a new set of joint future scenarios, re-optimize
    the remaining horizon, and execute only the next action.

For the first implementation, there is **no need for a separate
two-stage stochastic-programming formulation**. The Day-Ahead optimizer
and the rolling MPC can both use scenario-based stochastic optimization
directly.

------------------------------------------------------------------------

## 1. Deterministic baseline

The existing deterministic approach should remain as the baseline.

For a 24-hour horizon, forecasting models produce point forecasts such
as

\[ `\hat `{=tex}G\_{1:24} \]

for PV generation and

\[ `\hat `{=tex}P\_{1:24} \]

for market prices.

The optimizer then solves

\[ x\^\* = `\arg`{=tex}`\min`{=tex}\_x
C(x;`\hat `{=tex}G,`\hat `{=tex}P) \]

subject to the physical and operational constraints already defined in
the design document.

This baseline is important because every probabilistic extension should
ultimately be evaluated against the realized economic performance of
this deterministic strategy.

------------------------------------------------------------------------

## 2. Why probabilistic forecasting is needed

A point forecast treats one future trajectory as if it were known. In
reality, both PV generation and prices are uncertain.

At time (t), the relevant object is therefore not only

\[ `\hat `{=tex}G\_{t+1:T}, `\qquad `{=tex}`\hat `{=tex}P\_{t+1:T}, \]

but a predictive distribution over possible future trajectories:

\[ p(G\_{t+1:T},P\_{t+1:T}`\mid `{=tex}I_t), \]

where (I_t) is the information available at time (t).

The dependence between PV and prices matters. Regional solar conditions
affect both local PV production and aggregate renewable generation,
which in turn can affect wholesale prices. Weather and renewable
forecasts can explain much of this dependence, but the remaining
forecast errors do not have to be conditionally independent.

Therefore, independently sampling PV and price distributions,

\[ p(G`\mid `{=tex}X_G),p(P`\mid `{=tex}X_P), \]

can generate unrealistic combinations and destroy residual dependence.
Scenario generation should preserve both:

-   temporal dependence within each 24-hour trajectory;
-   cross-dependence between PV and price forecast errors.

------------------------------------------------------------------------

## 3. Forecasting approaches

The forecasting layer does not have to be extremely sophisticated
initially. With a relatively small dataset, robust tabular/time-series
models plus empirical uncertainty estimation are preferable to a large
multivariate neural model.

### 3.1 Point forecasting baseline

Use separate forecasting models for PV and market prices.

Possible models:

-   LightGBM;
-   CatBoost;
-   gradient boosting with lagged/time/weather features;
-   classical time-series models as additional benchmarks.

For PV, useful predictors may include:

-   irradiance / solar forecast;
-   cloud cover;
-   temperature;
-   hour of day;
-   day of year;
-   historical PV generation;
-   relevant lagged variables.

For prices, useful predictors may include:

-   hour/day/calendar variables;
-   lagged prices;
-   demand/load forecasts;
-   solar generation forecasts;
-   wind forecasts;
-   weather variables;
-   fuel or other market variables if available.

Importantly, solar/weather information should also enter the **price
model**. This allows the conditional price forecast to account for the
expected effect of high renewable production.

### 3.2 Quantile forecasting

A natural first probabilistic extension is quantile regression, for
example with LightGBM:

\[ Q\_{0.1}(Y`\mid `{=tex}X),`\quad`{=tex}
Q\_{0.5}(Y`\mid `{=tex}X),`\quad`{=tex} Q\_{0.9}(Y`\mid `{=tex}X). \]

A richer implementation could estimate

\[
Q\_{0.05},Q\_{0.10},Q\_{0.25},Q\_{0.50},Q\_{0.75},Q\_{0.90},Q\_{0.95}.
\]

Advantages:

-   easy to implement on top of the deterministic forecasting pipeline;
-   no assumption about a Gaussian conditional distribution;
-   directly provides uncertainty bands;
-   can be evaluated with pinball loss and empirical coverage;
-   useful for checking whether residual-based scenarios have plausible
    dispersion.

Quantile forecasts alone, however, do not define the joint 24-hour
PV/price trajectory distribution.

### 3.3 Distributional forecasting

Another option is to model parameters of a conditional distribution:

\[ Y`\mid `{=tex}X `\sim `{=tex}D(`\theta`{=tex}(X)). \]

For example,

\[ Y`\mid `{=tex}X
`\sim `{=tex}`\mathcal `{=tex}N(`\mu`{=tex}(X),`\sigma`{=tex}\^2(X)) \]

or a Student-t distribution for heavier tails.

This provides a smooth predictive distribution and straightforward
sampling. The disadvantage is distributional misspecification: a
convenient parametric family may not represent the actual conditional
error distribution well.

### 3.4 Explicit multivariate probabilistic forecasting --- later extension

A more advanced model could estimate directly

\[ p(G\_{1:24},P\_{1:24}`\mid `{=tex}X) \]

and sample coherent joint trajectories from it.

Possible approaches include autoregressive probabilistic models,
multivariate sequence models, copula-based models, normalizing flows,
and related methods.

This is theoretically attractive but should not be the initial
implementation because:

-   the target distribution is high-dimensional;
-   available historical data are limited;
-   the dependence structure is complex;
-   a sophisticated model may not outperform a well-designed empirical
    scenario generator.

The first version should therefore use simpler forecasting models and
construct the joint distribution empirically from out-of-sample residual
trajectories.

------------------------------------------------------------------------

## 4. Joint residual bootstrap: proposed stochastic baseline

### 4.1 Out-of-sample residuals

For each historical day (d), generate forecasts using only information
that would genuinely have been available at prediction time.

For PV:

\[ `\epsilon`{=tex}\^G_d = G\^{actual}*{d,1:24} -
`\hat `{=tex}G*{d,1:24}. \]

For price:

\[ `\epsilon`{=tex}\^P_d = P\^{actual}*{d,1:24} -
`\hat `{=tex}P*{d,1:24}. \]

Store them jointly:

\[ R_d = (`\epsilon`{=tex}^G\_{d,1:24},`\epsilon`{=tex}^P\_{d,1:24}). \]

Thus one historical residual block contains a complete 48-dimensional
error trajectory.

The residuals used for scenario generation must be **out-of-sample /
rolling-origin residuals**, not in-sample training residuals. Otherwise
uncertainty will be systematically underestimated.

### 4.2 Scenario generation

Suppose today's point forecasts are

\[ `\hat `{=tex}G\_{1:24},`\qquad`{=tex} `\hat `{=tex}P\_{1:24}. \]

Sample a historical residual block (R_d) and construct:

\[ G\^{(s)}*{1:24} = `\hat `{=tex}G*{1:24} +
`\epsilon`{=tex}\^{G,(s)}\_{1:24}, \]

\[ P\^{(s)}*{1:24} = `\hat `{=tex}P*{1:24} +
`\epsilon`{=tex}\^{P,(s)}\_{1:24}. \]

Repeat for

\[ s=1,`\ldots`{=tex},S, \]

for example (S=100), (500), or (1000).

Each scenario is therefore

\[ `\xi`{=tex}\^{(s)} = (G\^{(s)}*{1:24},P\^{(s)}*{1:24}). \]

Sampling the entire PV/price residual block together preserves
empirically observed:

-   hour-to-hour PV error dependence;
-   hour-to-hour price error dependence;
-   dependence between PV and price errors;
-   realistic daily error shapes.

This avoids assuming that 48 uncertain quantities are independent.

### 4.3 Physical corrections

Generated scenarios may require simple physical corrections, for
example:

\[ G_t\^{(s)} `\ge 0`{=tex}. \]

PV generation should also respect plant capacity and nighttime
zero-generation constraints.

Care should be taken not to apply arbitrary corrections that distort the
empirical error distribution.

------------------------------------------------------------------------

## 5. Scenario-based stochastic optimization

Instead of optimizing one forecast trajectory, optimize a decision
against the generated scenario set.

For equal-probability scenarios:

\[ `\min`{=tex}*x `\frac{1}{S}`{=tex} `\sum`{=tex}*{s=1}\^{S}
C(x,`\xi`{=tex}\^{(s)}) \]

subject to the relevant physical and operational constraints.

This is a Sample Average Approximation (SAA) of

\[ `\min`{=tex}\_x E\[C(x,`\Xi`{=tex})\]. \]

The exact decision vector (x) should follow the operational variables
already defined in the main design document: load scheduling, battery
charge/discharge, market purchases/sales, etc.

For a modest number of hourly variables and a few hundred scenarios,
computational time should not be a primary constraint. The operational
setting provides substantially more time than the expected LP/MILP
solution time.

------------------------------------------------------------------------

## 6. Risk-aware objective: CVaR extension

Expected cost alone can hide rare but expensive outcomes. A useful
extension is

\[ `\min`{=tex}*x E\[C(x,`\Xi`{=tex})\] +
`\lambda`{=tex},CVaR*{`\alpha`{=tex}}(C(x,`\Xi`{=tex})). \]

For example,

\[ `\alpha=0.95`{=tex}. \]

CVaR represents the average cost in the worst tail of scenarios.

The standard linear representation is

\[ CVaR\_`\alpha`{=tex}(C) = `\eta`{=tex}+ `\frac{1}{(1-\alpha)S}`{=tex}
`\sum`{=tex}\_{s=1}\^{S}z_s, \]

with

\[ z_s`\ge `{=tex}C_s-`\eta`{=tex}, `\qquad`{=tex} z_s`\ge0`{=tex}. \]

Therefore CVaR can be incorporated into an LP/MILP without turning the
optimization problem into a nonlinear stochastic solver.

The parameter (`\lambda`{=tex}) controls the trade-off between average
cost and protection against expensive tail scenarios.

------------------------------------------------------------------------

## 7. Day-Ahead workflow

At the Day-Ahead decision time:

1.  Collect the current information set (I_0).
2.  Generate PV and price forecasts.
3.  Generate joint PV/price scenarios using the residual bootstrap.
4.  Solve the scenario-based stochastic optimization problem.
5.  Submit the required Day-Ahead decision/schedule.
6.  Store the forecasts, scenarios, decisions, and later realized
    outcomes for backtesting.

Conceptually:

\[ I_0 `\rightarrow`{=tex} (`\hat `{=tex}G,`\hat `{=tex}P)
`\rightarrow`{=tex} {`\xi`{=tex}^{(1)},`\ldots`{=tex},`\xi`{=tex}^{(S)}}
`\rightarrow`{=tex} `\text{stochastic optimizer}`{=tex}
`\rightarrow`{=tex} `\text{Day-Ahead decision}`{=tex}. \]

No separate two-stage stochastic-programming layer is required in the
initial implementation.

------------------------------------------------------------------------

## 8. Intraday / MPC workflow

The same architecture is reused during operation.

At time (t), additional information is available:

-   actual PV generation up to (t);
-   observed market prices;
-   current battery SOC;
-   updated weather forecast;
-   updated load information;
-   other newly observed system variables.

Define the updated information set (I_t).

Forecast only the remaining horizon:

\[ `\hat `{=tex}G\_{t+1:T`\mid `{=tex}t}, `\qquad`{=tex}
`\hat `{=tex}P\_{t+1:T`\mid `{=tex}t}. \]

Generate new joint scenarios:

\[ `\xi`{=tex}\^{(1)}*{t+1:T},`\ldots`{=tex},`\xi`{=tex}\^{(S)}*{t+1:T}.
\]

Solve the optimization for the remaining horizon subject to already
fixed commitments and the current physical state.

Execute only the next control action.

Then repeat:

\[ `\boxed{
observe
\rightarrow
forecast
\rightarrow
joint\ scenarios
\rightarrow
optimize
\rightarrow
execute\ next\ action
\rightarrow
observe
}`{=tex} \]

This is the proposed stochastic MPC / rolling-horizon architecture.

The conceptual difference between Day Ahead and MPC is therefore
primarily the **information set and remaining horizon**, not a
fundamentally different optimization framework.

------------------------------------------------------------------------

## 9. Improved residual scenario generation

Simple unconditional joint residual bootstrap is the baseline. Several
extensions should be considered.

### 9.1 Conditional / stratified residual bootstrap

Forecast errors may depend on the operating regime.

Instead of sampling

\[ R_d`\sim`{=tex}`\hat `{=tex}P(R), \]

sample approximately from

\[ R_d`\sim`{=tex}`\hat `{=tex}P(R`\mid `{=tex}Z), \]

where (Z) describes the current forecasting situation.

Possible conditioning variables:

-   season/month;
-   predicted solar generation;
-   irradiance/cloudiness regime;
-   predicted price level;
-   wind regime;
-   expected system load;
-   weekday/weekend.

With limited data, conditioning must remain coarse enough to retain a
reasonable number of historical observations.

### 9.2 k-nearest-neighbour residual bootstrap

Rather than defining discrete regimes, construct a feature vector
describing today's situation:

\[ Z_0=(weather, solar, wind, demand, season,`\ldots`{=tex}). \]

Find the (k) most similar historical forecasting situations and sample
joint residual blocks only from these neighbours.

This approximates

\[ p(`\epsilon`{=tex}^G,`\epsilon`{=tex}^P`\mid `{=tex}Z_0) \]

without fitting a high-dimensional parametric distribution.

This is a particularly attractive extension for a small-to-medium
dataset.

### 9.3 Residual scaling

Forecast uncertainty may be heteroscedastic.

For example, PV uncertainty at midday can be much larger than at night,
and price uncertainty can depend strongly on the expected price regime.

One possibility is to model residual scale:

\[ `\epsilon`{=tex}\_t=`\sigma`{=tex}(X_t)z_t \]

and bootstrap standardized residuals (z_t), then rescale them using the
current predicted uncertainty.

This can make historical residual trajectories more transferable across
different operating regimes.

### 9.4 Residual diagnostics

Before relying on bootstrap scenarios, examine:

-   residual distributions by hour;
-   residual variance by predicted value;
-   autocorrelation within PV residuals;
-   autocorrelation within price residuals;
-   contemporaneous correlation between PV and price residuals;
-   cross-correlation

\[ Corr(`\epsilon`{=tex}^G_t,`\epsilon`{=tex}^P\_{t+k}); \]

-   seasonal changes in residual structure;
-   extreme-error behaviour;
-   calibration of generated scenario intervals against realized
    outcomes.

A useful empirical question is:

> How much PV-price dependence is already explained by common predictors
> such as weather and renewable forecasts, and how much dependence
> remains in the joint residual structure?

------------------------------------------------------------------------

## 10. Alternative uncertainty and optimization approaches

The proposed joint-bootstrap + SAA approach should be treated as the
primary stochastic baseline, not the only possible method.

### 10.1 Robust optimization

Instead of estimating probabilities, define an uncertainty set:

\[ `\Xi`{=tex}`\in`{=tex}`\mathcal `{=tex}U \]

and solve

\[
`\min`{=tex}*x`\max`{=tex}*{`\xi`{=tex}`\in`{=tex}`\mathcal `{=tex}U}C(x,`\xi`{=tex}).
\]

Advantages:

-   does not require an accurately estimated probability distribution;
-   provides protection against adverse realizations.

Disadvantage:

-   may be overly conservative, especially if the uncertainty set is
    broad.

This can be useful as a benchmark.

### 10.2 Distributionally Robust Optimization (DRO)

DRO is particularly interesting when the historical dataset is limited.

Instead of assuming that the empirical scenario distribution
(`\hat `{=tex}P) is the true distribution, define an ambiguity set of
plausible distributions around it:

\[ `\mathcal `{=tex}P =
{P:d(P,`\hat `{=tex}P)`\le`{=tex}`\epsilon`{=tex}}. \]

Then solve

\[ `\boxed{
\min_x
\max_{P\in\mathcal P}
E_P[C(x,\Xi)]
}`{=tex} \]

The interpretation is:

> optimize against the worst plausible probability distribution near the
> empirical distribution, rather than against one estimated distribution
> or against the single worst physical scenario.

Potential ambiguity-set constructions include Wasserstein-distance and
moment-based sets.

DRO is attractive for this project because it directly addresses
**distribution-estimation uncertainty caused by limited data**.

It should be considered a later extension after the SAA baseline is
working.

### 10.3 Copula-based scenario generation

Estimate marginal predictive distributions for PV and price and connect
them through a copula.

Conceptually:

\[ F\_{G,P}(g,p) = C(F_G(g),F_P(p)). \]

This can model dependence more explicitly than independent marginals,
but extending the approach to coherent 24-hour PV/price trajectories
creates a high-dimensional dependence problem.

It is therefore better treated as an experimental extension.

### 10.4 Explicit multivariate probabilistic models

A model can attempt to learn directly

\[ p(G\_{1:24},P\_{1:24}`\mid `{=tex}X). \]

This avoids a separate residual-bootstrap layer and can generate joint
trajectories directly.

However, because the dataset is limited and the output distribution is
high-dimensional, this approach has substantial model-risk and
overfitting risk. It should be evaluated only after establishing strong
simpler baselines.

### 10.5 Conformal uncertainty estimation

Conformal methods can provide empirically calibrated prediction
intervals or sets without requiring a correctly specified parametric
distribution.

They may be useful for:

-   checking/calibrating forecast uncertainty;
-   defining uncertainty bounds;
-   supporting chance constraints.

They do not by themselves solve the problem of generating coherent joint
24-hour PV/price trajectories, so they should complement rather than
replace the scenario generator.

### 10.6 Decision-focused / predict-then-optimize learning

The standard project architecture trains forecasting models using
statistical losses such as MAE, RMSE, pinball loss, or likelihood and
evaluates the resulting decisions separately.

A later research extension could instead train models with downstream
optimization performance in mind:

\[ `\theta`{=tex}\^\* = `\arg`{=tex}`\min`{=tex}\_`\theta`{=tex}
C(x\^\*(f\_`\theta`{=tex}(X)),Y). \]

The key idea is that not all forecast errors have equal economic
consequences.

This is interesting but substantially increases system complexity and is
not recommended for the initial implementation.

------------------------------------------------------------------------

## 11. Evaluation

Forecast quality and decision quality should be evaluated separately.

### Forecasting metrics

For point forecasts:

-   MAE;
-   RMSE.

For probabilistic forecasts:

-   pinball loss;
-   interval coverage;
-   interval width;
-   CRPS where appropriate;
-   calibration plots.

For scenario generation:

-   marginal calibration;
-   temporal correlation reproduction;
-   PV-price residual correlation reproduction;
-   tail behaviour;
-   realism of complete trajectories.

### Optimization metrics

The primary metric should be **realized out-of-sample economic cost**,
not forecasting error alone.

Compare at least:

1.  deterministic point-forecast optimization;
2.  joint-bootstrap SAA;
3.  conditional/KNN joint-bootstrap SAA;
4.  SAA + CVaR.

Possible additional benchmarks:

5.  robust optimization;
6.  DRO;
7.  day-ahead Oracle with factual PV and day-ahead prices, one physical schedule, and zero deliberate imbalance.

Additional operational metrics may include:

-   mean realized cost;
-   median cost;
-   P95 / tail cost;
-   worst 5% average cost;
-   battery cycling/degradation;
-   energy bought/sold;
-   PV curtailment/export;
-   constraint violations;
-   deviation from Day-Ahead commitments.

A central project question should be:

> Does better representation of uncertainty improve realized decisions
> and economic outcomes, even when point-forecast accuracy changes
> little?

------------------------------------------------------------------------

## 12. Recommended implementation sequence

A practical sequence is:

1.  **Deterministic baseline**
    -   point PV forecast;
    -   point price forecast;
    -   existing LP/MILP.
2.  **Probabilistic forecast diagnostics**
    -   quantile LightGBM/CatBoost;
    -   rolling out-of-sample residual collection;
    -   calibration analysis.
3.  **Joint residual bootstrap**
    -   sample complete PV + price daily residual blocks;
    -   generate 100-1000 coherent scenarios.
4.  **SAA stochastic optimizer**
    -   minimize average scenario cost;
    -   compare with deterministic optimization.
5.  **Rolling stochastic MPC**
    -   update forecasts and scenarios intraday;
    -   re-optimize remaining horizon;
    -   execute only the next action.
6.  **Conditional scenario generation**
    -   stratified bootstrap or k-nearest-neighbour residual bootstrap;
    -   optional heteroscedastic residual scaling.
7.  **Risk-aware optimization**
    -   add CVaR;
    -   evaluate expected-cost versus tail-risk trade-off.
8.  **Research extensions**
    -   DRO;
    -   copulas;
    -   explicit multivariate probabilistic forecasting;
    -   conformal calibration;
    -   decision-focused learning.

------------------------------------------------------------------------

## 13. Key design decision to preserve

The initial probabilistic implementation should remain intentionally
modular:

\[ `\boxed{
\text{Forecasting}
\rightarrow
\text{Joint scenario generation}
\rightarrow
\text{Optimization}
}`{=tex} \]

This makes it possible to replace one component at a time and measure
its incremental value.

For example:

\[ `\text{LightGBM + bootstrap + SAA}`{=tex} \]

can later be compared with

\[ `\text{LightGBM + conditional bootstrap + SAA}`{=tex}, \]

\[ `\text{probabilistic multivariate model + SAA}`{=tex}, \]

or

\[ `\text{empirical scenarios + DRO}`{=tex}. \]

This modularity is especially valuable for a portfolio/research project
because it allows the economic value of each methodological upgrade to
be measured independently rather than hiding all improvements inside one
complex end-to-end model.
