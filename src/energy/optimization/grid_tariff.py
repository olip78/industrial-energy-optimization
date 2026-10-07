"""Regulated grid charges used by the economic replay.

The wholesale market settles signed commercial positions.  Grid tariffs are
different: they apply to positive physical withdrawal from the public grid and
are not earned back when electricity is exported later.  Keeping this contract
separate prevents a signed day-ahead position from accidentally netting grid
charges across hours.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridTariffParameters:
    """Variable import charges and fixed annual charges for one meter.

    All variable values are expressed in EUR/kWh and exclude VAT.  The daytime
    network rate is used for the 06:00--22:00 optimization window.  The
    nighttime rate enters the conservative cost of replenishing battery energy
    bought during the previous night.
    """

    daytime_network_eur_per_kwh: float = 0.0
    nighttime_network_eur_per_kwh: float = 0.0
    levies_eur_per_kwh: float = 0.0
    concession_fee_eur_per_kwh: float = 0.0
    electricity_tax_eur_per_kwh: float = 0.0
    export_fee_eur_per_kwh: float = 0.0
    annual_network_base_eur: float = 0.0
    annual_metering_eur: float = 0.0
    billing_days_per_year: int = 365
    name: str = "no_grid_tariff"

    def __post_init__(self) -> None:
        non_negative = (
            self.daytime_network_eur_per_kwh,
            self.nighttime_network_eur_per_kwh,
            self.levies_eur_per_kwh,
            self.concession_fee_eur_per_kwh,
            self.electricity_tax_eur_per_kwh,
            self.export_fee_eur_per_kwh,
            self.annual_network_base_eur,
            self.annual_metering_eur,
        )
        if any(value < 0 for value in non_negative):
            raise ValueError("grid tariff values must be non-negative")
        if self.billing_days_per_year <= 0:
            raise ValueError("billing_days_per_year must be positive")

    @property
    def daytime_variable_import_eur_per_kwh(self) -> float:
        return (
            self.daytime_network_eur_per_kwh
            + self.levies_eur_per_kwh
            + self.concession_fee_eur_per_kwh
            + self.electricity_tax_eur_per_kwh
        )

    @property
    def nighttime_variable_import_eur_per_kwh(self) -> float:
        return (
            self.nighttime_network_eur_per_kwh
            + self.levies_eur_per_kwh
            + self.concession_fee_eur_per_kwh
            + self.electricity_tax_eur_per_kwh
        )

    @property
    def fixed_daily_eur(self) -> float:
        return (
            self.annual_network_base_eur + self.annual_metering_eur
        ) / self.billing_days_per_year

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "daytime_network_eur_per_kwh": self.daytime_network_eur_per_kwh,
            "nighttime_network_eur_per_kwh": self.nighttime_network_eur_per_kwh,
            "levies_eur_per_kwh": self.levies_eur_per_kwh,
            "concession_fee_eur_per_kwh": self.concession_fee_eur_per_kwh,
            "electricity_tax_eur_per_kwh": self.electricity_tax_eur_per_kwh,
            "export_fee_eur_per_kwh": self.export_fee_eur_per_kwh,
            "annual_network_base_eur": self.annual_network_base_eur,
            "annual_metering_eur": self.annual_metering_eur,
            "billing_days_per_year": self.billing_days_per_year,
            "daytime_variable_import_eur_per_kwh": (
                self.daytime_variable_import_eur_per_kwh
            ),
            "nighttime_variable_import_eur_per_kwh": (
                self.nighttime_variable_import_eur_per_kwh
            ),
            "fixed_daily_eur": self.fixed_daily_eur,
        }


NO_GRID_TARIFF = GridTariffParameters()


# Reference site: approximate Pforzheim PV location already used by the data
# layer.  SWP 2025 SLP low-voltage tariff: 5.49 ct/kWh at 06:00--22:00,
# 2.75 ct/kWh at night, EUR 80/a base charge and EUR 29.21/a for a bidirectional
# meter.  Nationwide 2025 non-privileged levies are 0.277 ct/kWh KWKG,
# 1.558 ct/kWh special grid usage and 0.816 ct/kWh offshore.  Pforzheim falls
# in the <=500,000 population concession bracket (1.99 ct/kWh).  The fictional
# plant is assumed eligible for the producing-industry StromStG section 9b
# relief, leaving 0.05 ct/kWh after application.
PFORZHEIM_SLP_2025_GRID_TARIFF = GridTariffParameters(
    daytime_network_eur_per_kwh=0.0549,
    nighttime_network_eur_per_kwh=0.0275,
    levies_eur_per_kwh=0.02651,
    concession_fee_eur_per_kwh=0.0199,
    electricity_tax_eur_per_kwh=0.0005,
    export_fee_eur_per_kwh=0.0,
    annual_network_base_eur=80.0,
    annual_metering_eur=29.21,
    billing_days_per_year=365,
    name="pforzheim_slp_2025_producing_industry",
)
