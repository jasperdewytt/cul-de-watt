"""Compact final-settlement PLAN metrics and fixed-dispatch figures."""
from pathlib import Path
import numpy as np
import pandas as pd


def dispersion(gains, reference):
    gains, reference = np.asarray(gains), np.asarray(reference)
    percentages = np.divide(gains,reference,out=np.full_like(gains,np.nan),where=reference>0)
    total = gains.sum()
    return {
        "minimum_gain_aud":float(gains.min()), "gain_range_aud":float(np.ptp(gains)),
        "gain_gini":float(np.abs(gains[:,None]-gains).sum()/(2*len(gains)*total)) if (gains>=0).all() and total>0 else np.nan,
        "positive_savings_percentage_ratio":float(percentages.max()/percentages.min()) if np.isfinite(percentages).all() and (percentages>0).all() else np.nan,
    }


def write_headlines(output, tables, summary, accounts):
    physical = ("demand_kwh","pv_kwh","self_consumption","self_sufficiency","import_kwh","export_kwh",
                "peak_import_kw","peak_export_kw","peak_period_import_kwh","discharge_kwh","equivalent_full_cycles")
    community, household = [], []
    for entry in summary.to_dict("records"):
        label, fee, scenario = entry["scenario"], entry["fee_aud_per_member_year"], entry["physical_scenario"]
        totals = tables[scenario]["totals"]
        a = accounts[(accounts.scenario==label)&(accounts.fee_aud_per_member_year==fee)].sort_values("member")
        row = {"scenario":label,"fee_aud_per_member_year":fee,"physical_scenario":scenario,
               "settlement_status":entry["status"],"actual_budget_aud":entry["actual_budget_aud"],
               **{k:totals.get(k,np.nan) for k in physical},"emissions_status":"Unavailable: no sourced NSW factor"}
        if len(a)==9 and a.final_net_economic_cost_aud.notna().all():
            bills=a.final_net_economic_cost_aud.to_numpy()
            row.update(electricity_bill_aud=float(a.final_electricity_bill_aud.sum()),net_economic_cost_aud=float(bills.sum()),
                       battery_wear_aud=float(a.battery_wear_aud.sum()),operating_cost_aud=float(a.operating_cost_total_aud.sum()),
                       oracle_gap_aud=float(bills.sum()-tables["cooperative_oracle"]["totals"]["net_economic_cost_aud"]-9*fee))
            for baseline in ("independent_rule","independent_oracle"):
                reference=tables[baseline]["household"].loc[a.member,"net_economic_cost_aud"].to_numpy()
                gains=reference-bills
                row[f"savings_vs_{baseline}_aud"]=float(gains.sum())
                row[f"savings_vs_{baseline}_fraction"]=float(gains.sum()/reference.sum()) if reference.sum()>0 else np.nan
                row.update({f"{k}_vs_{baseline}":v for k,v in dispersion(gains,reference).items()})
            for record in a.to_dict("records"):
                h={"scenario":label,"fee_aud_per_member_year":fee,"member":record["member"],"physical_scenario":scenario,
                   **{k:record.get(k,np.nan) for k in physical},
                   "electricity_bill_aud":record["final_electricity_bill_aud"],"net_economic_cost_aud":record["final_net_economic_cost_aud"],
                   "battery_wear_aud":record["battery_wear_aud"],"operating_cost_aud":record["operating_cost_total_aud"]}
                for baseline in ("independent_rule","independent_oracle"):
                    r=float(tables[baseline]["household"].loc[record["member"],"net_economic_cost_aud"])
                    h[f"gain_vs_{baseline}_aud"]=r-h["net_economic_cost_aud"]
                    h[f"gain_vs_{baseline}_fraction"]=(r-h["net_economic_cost_aud"])/r if r>0 else np.nan
                household.append(h)
        community.append(row)
    pd.DataFrame(community).to_csv(output/"headline_metrics.csv",index=False,float_format="%.10f")
    pd.DataFrame(household).to_csv(output/"household_headlines.csv",index=False,float_format="%.10f")
    lines=["# Stage 04 headline metrics","","Every scenario has a row in headline_metrics.csv; household_headlines.csv supplies H metrics.",
           "Physical metrics retain each scenario's Stage 02 dispatch. Settlement changes distribution only.",
           "Infeasible settlements have no final bills. Gini requires nonnegative gains and positive total; the percentage ratio requires all positive defined percentages.",
           "Cash electricity excludes wear and mechanism operating costs. Emissions remain unavailable. Oracle gaps match coordinator fees.","",
           "| Scenario | Fee/member | Net cost | Minimum gain vs oracle | Maximum excess |","|---|---:|---:|---:|---:|"]
    for record in summary.to_dict("records"):
        def number(key):
            v=record.get(key,np.nan)
            return f"{v:.2f}" if pd.notna(v) else "Unavailable"
        lines.append(f"| {record['scenario']} | {record['fee_aud_per_member_year']:.0f} | {number('allocated_total_aud')} | {number('min_gain_vs_independent_oracle_aud')} | {number('max_positive_excess_aud')} |")
    (output/"headline_metrics.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    explanation=["# Nine household bill explanations", "", "AUD/year. Grid cash = import charges minus feed-in revenue plus connection fees. Internal payments and settlement transfers cancel. Wear is economic, not a cash payment.", "",
                 "The oracle nucleolus is shown as a stable comparator; the sharing income-only dial-1 example protects singletons but fails the subgroup stress test. Neither is a monthly guarantee."]
    for label in ("nucleolus","target_B_independent_oracle_individual_inverse_income_lambda_1"):
        explanation += ["",f"## {label}",""]
        selected=accounts[(accounts.scenario==label)&(accounts.fee_aud_per_member_year==0)].sort_values("member")
        for row in selected.itertuples():
            outside=float(tables["independent_oracle"]["household"].loc[row.member,"net_economic_cost_aud"])
            grid=row.import_charge_aud-row.feedin_revenue_aud+row.connection_fee_aud
            explanation.append(f"Member {row.member}: strong outside option **{outside:.2f}**. Grid cash {grid:.2f}, net internal payment {row.internal_payment_aud:.2f}, battery wear {row.battery_wear_aud:.2f}, and base operating cost {row.operating_cost_once_aud:.2f} give the physical account {row.physical_account_net_economic_cost_aud:.2f}. The settlement transfer is {row.settlement_adjustment_aud:+.2f}; the additional coordinator fee is {row.coordinator_fee_aud:.2f}. Final electricity cash is **{row.final_electricity_bill_aud:.2f}**, and wear-inclusive net cost is **{row.final_net_economic_cost_aud:.2f}**, a gain of **{row.gain_vs_independent_oracle_aud:.2f}** against the strong outside option. A negative bill is a credit, not extra community value.")
            explanation.append("")
    (output/"household_explanations.md").write_text("\n".join(explanation).rstrip()+"\n",encoding="utf-8")



def write_figures(output, summary, accounts, ablation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen=summary[(summary.fee_aud_per_member_year==0)&summary.scenario.str.startswith("oracle_need_")]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode in ("equal","inverse_income","occupants_income"):
        rows=chosen[chosen.scenario.str.startswith(f"oracle_need_{mode}_lambda_")].copy()
        rows["dial"]=rows.scenario.str.rsplit("_",n=1).str[-1].astype(float)
        rows=rows.sort_values("dial")
        m5=accounts[(accounts.member==5)&(accounts.fee_aud_per_member_year==0)].set_index("scenario")
        axes[0].plot(rows.dial,[m5.loc[x,"gain_vs_independent_oracle_aud"] for x in rows.scenario],marker="o",label=mode)
        axes[1].plot(rows.dial,rows.l1_distortion_aud,marker="o",label=mode)
    axes[0].set(xlabel="Need dial",ylabel="Member 5 annual gain (AUD)",title="Oracle: core-constrained redistribution")
    axes[1].set(xlabel="Need dial",ylabel="L1 target distortion (AUD)",title="Distance from the requested distribution")
    axes[0].legend();fig.tight_layout();fig.savefig(output/"need_frontier.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    subset=summary[(summary.scenario=="sharing_midpoint_actual")].sort_values("fee_aud_per_member_year")
    x=np.arange(len(subset));ax.bar(x-.18,subset.actual_budget_aud,.36,label="Actual sharing cost")
    ax.bar(x+.18,subset.max_fundable_budget_aud,.36,label="Strong-bounds funding ceiling")
    ax.set(xticks=x,xticklabels=[f"{v:g}" for v in subset.fee_aud_per_member_year],xlabel="Illustrative coordinator fee (AUD/member/year)",ylabel="Annual community cost (AUD)",title="Sharing cannot fund all strong subgroup bounds")
    ax.legend();fig.tight_layout();fig.savefig(output/"sharing_funding.png",dpi=140);plt.close(fig)
    return ["need_frontier.png","sharing_funding.png"]
