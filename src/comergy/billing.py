"""External cash bill and economic costs, with no internal settlement yet."""
import numpy as np


def bill_household(hourly, import_price, export_price, *, days, daily_fee,
                   wear_rate, operating_cost):
    charges = float(np.dot(hourly.import_kwh, import_price))
    revenue = float(np.dot(hourly.export_kwh, export_price))
    fees = days * daily_fee
    bill = charges - revenue + fees
    wear = float(hourly.discharge_kwh.sum()) * wear_rate
    return dict(import_charge_aud=charges, feedin_revenue_aud=revenue,
                connection_fee_aud=fees, internal_payment_aud=0.0,
                electricity_bill_aud=bill, battery_wear_aud=wear,
                operating_cost_aud=operating_cost,
                net_economic_cost_aud=bill + wear + operating_cost)
