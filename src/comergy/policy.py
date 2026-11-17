"""Causal policies observe only current meter energy and stored usable energy."""


def self_consumption(demand, pv, soc, battery, dt):
    """Return requested AC charge/discharge kWh, own PV before grid exchange."""
    surplus = pv - demand
    if surplus >= 0:
        return min(surplus, battery.power_kw * dt,
                   max(0, battery.capacity_kwh - soc) / battery.eta_charge), 0.0
    return 0.0, min(-surplus, battery.power_kw * dt, soc * battery.eta_discharge)
