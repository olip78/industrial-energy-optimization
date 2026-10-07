# PV source-time hypothesis test

## Result

Use **Europe/Berlin as the provisional working convention** for the PV source clock.

The evidence is mixed if we look only at one generic weather-regression score: profile 1a favours the UTC control (R² 0.576) over Berlin (R² 0.312). But its morning generation starts at the same hour as local solar radiation under Berlin, and its generation ends three hours earlier, a plausible pattern for the documented southeast-facing panel. Profiles 2a and 2b favour Berlin on both weather-fit and onset diagnostics. Under Berlin, all three profiles have a median PV-start minus weather-start lag of 0 hours on the top 10% highest-radiation days.

This is enough to choose a transparent modelling convention for Version 1. It does not replace confirmation by the source publisher.

## Method

1. Aggregate measured PV profiles 1a, 2a and 2b from 15 minutes to hourly mean power.
2. Exclude the three dates that are zero across all three profiles.
3. Interpret the same naïve `source_time` under each timezone hypothesis, convert it to UTC, and join it to ERA5 weather at the approximate Pforzheim point.
4. Compare daylight PV output with global, direct and diffuse solar radiation, temperature and wind through a simple least-squares model. Open-Meteo reports radiation averaged over the preceding hour, so weather is shifted one label back to match PV's timestamp hour.

The orientation of the 1a panels is southeast. This is why the test does not compare the hour of daily PV maximum with the hour of maximum horizontal radiation directly.

## Decision and output

`pv_power_15min_assumed_europe_berlin.csv.gz` is a new join-ready copy. It preserves `source_time`, adds `valid_time_utc`, and labels the assumption. It also labels the three suspected common outage dates. The raw table remains unchanged.

The source emits 96 rows on daylight-saving transition dates, so four rows on each spring/autumn transition cannot be mapped unambiguously to UTC. Their `valid_time_utc` is deliberately blank and `timezone_conversion_status` is `unresolved_dst_transition`; do not silently fill them.

Detailed figures are in `data/processed/pv_timezone_hypothesis_metrics.csv` and `data/metadata/pv_timezone_hypothesis_metrics.json`.
