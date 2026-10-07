# Day-ahead electricity price forecasting: data ideas

## Goal

Build a reproducible pet-project dataset for forecasting German
day-ahead electricity prices over a long historical period
(approximately 2005/2006--2025).

The emphasis is on data that are reasonably easy to obtain and can be
made consistent over a long backtest, rather than reconstructing a
production-grade market forecasting system.

## Suggested feature groups

### 1. Historical electricity prices

Use the target series itself to construct autoregressive and seasonal
features:

-   price at the same delivery hour 1 and 2 days ago;
-   price at the same hour 1, 2, 3 and 4 weeks ago;
-   optionally, the full previous-day price curve;
-   rolling statistics and relative price-to-daily-mean features.

Example:

`price_lag_1d`, `price_lag_2d`, `price_lag_7d`, `price_lag_14d`,
`price_lag_21d`, `price_lag_28d`.

### 2. Calendar features

-   delivery hour;
-   day of week;
-   weekend;
-   German public holiday;
-   month / season;
-   cyclic hour and day-of-year features;
-   correct handling of DST (23/25-hour days).

### 3. Weather over Germany

For the price model, local weather at the household/PV location is not
sufficient. Use spatial weather information over Germany, for example
several representative regions or grid points:

-   temperature;
-   wind speed;
-   shortwave / solar radiation;
-   cloud cover.

Possible representations:

-   raw values at 8--12 representative locations;
-   regional averages;
-   PCA / latent weather factors (e.g. overall German wind level,
    north--south gradient).

**Important:** ERA5 is reanalysis, not the weather forecast that market
participants actually knew day-ahead. Using future-hour ERA5 values
therefore represents a **perfect-weather / weather-oracle experiment**,
not a strictly ex-ante forecast.

### 4. Load

If available, include German electricity load and lagged load.

A more realistic day-ahead experiment can also train a separate load
model from calendar + temperature and feed its forecast into the price
model.

### 5. Renewable-generation proxies

Instead of relying only on raw weather, construct physically meaningful
proxies:

`solar_potential = solar/radiation_factor × installed_PV_capacity`

`wind_potential = wind_capacity_factor × installed_wind_capacity`

This is important over a 20-year sample because identical weather had a
very different market impact in 2006 and 2025 due to the large increase
in installed renewable capacity.

A useful derived feature is approximate residual load:

`residual_load = load - wind_potential - solar_potential`

### 6. Optional extensions

If they are easy to obtain consistently:

-   historical wind forecast / generation;
-   installed generation capacity by technology;
-   gas price;
-   EUA/CO₂ price;
-   coal price;
-   hydro / pumped-storage information.

For the first pet-project version these are optional; price history +
calendar + weather + load + renewable-capacity interaction is sufficient
for a meaningful experiment.

## Data sources

  --------------------------------------------------------------------------
  Source            Data              Approximate       Notes
                                      useful period     
  ----------------- ----------------- ----------------- --------------------
  **Open Power      German            2005+             Good starting point
  System Data       electricity                         for the target
  (OPSD)**          prices                              series and
                                                        reproducible
                                                        processing

  **OPSD Time       German load and   load roughly      Convenient unified
  Series**          power-system time 2006+; other      hourly dataset
                    series            series vary       

  **Legacy OPSD     German wind       evidence of data  Worth investigating
  sources**         forecast / wind   from early 2005   as a genuine ex-ante
                    generation                          forecast feature

  **OPSD Generation Installed PV,     historical annual Useful for weather ×
  Capacity**        wind and other    series            installed-capacity
                    generation                          features
                    capacity                            

  **Copernicus      Hourly            1940+             Excellent consistent
  ERA5**            temperature,                        historical weather;
                    wind, radiation,                    reanalysis implies
                    cloud cover                         oracle-weather
                                                        caveat

  **ENTSO-E         Load, generation  coverage depends  Useful as an
  historical        and other system  on dataset;       extension, but less
  statistics /      data              modern            convenient for a
  Transparency                        Transparency data uniform 2005+
  data**                              mostly later      dataset

  **EEX /           German day-ahead  historical        Potential
  market-data       prices                              alternative/check
  sources**                                             for the OPSD price
                                                        series
  --------------------------------------------------------------------------

## Proposed first experiment

Use **2006--2025** as the main period and compare three increasingly
rich models:

1.  **Price + calendar**\
    Historical price lags and calendar features.

2.  **Price + calendar + ERA5**\
    Add spatial temperature, wind, radiation and cloud information.

3.  **Price + calendar + physical/system features**\
    Add load, installed PV/wind capacity, weather-derived renewable
    potential and approximate residual load.

This gives a simple ablation study:

`autoregressive → + weather → + energy-system structure`

For a later, stricter ex-ante version, replace ERA5 future weather with
archived weather forecasts where feasible and add genuine historical
wind/load forecasts.
