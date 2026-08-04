# csv_file = "./ztParameterStudy4/parameter_study_results.csv"      # <-- change this

#!/usr/bin/env python3
"""
Parameter sweep analysis.

Usage:
    python parameter_sweep_analysis.py input.csv

Outputs are written to analysis_results/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUTDIR="./ztParameterStudy6/parameter_study_analysis"

def ensure(p):
    os.makedirs(p, exist_ok=True)

def heatmap(df, metric, title, outpath, diverging=False, center=0.0, vmin=None, vmax=None):
    pivot=df.pivot(index="h_b", columns="normalizing_factor", values=metric)
    fig,ax=plt.subplots(figsize=(7,6))
    if diverging:
        if vmin is None: vmin=np.nanmin(pivot.values)
        if vmax is None: vmax=np.nanmax(pivot.values)
        m=max(abs(vmin-center),abs(vmax-center))
        norm=TwoSlopeNorm(vmin=center-m,vcenter=center,vmax=center+m)
        im=ax.imshow(pivot.values,origin="lower",aspect="auto",cmap="coolwarm",norm=norm)
    else:
        im=ax.imshow(pivot.values,origin="lower",aspect="auto",cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns],rotation=45,ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])
    ax.set_xlabel("normalizing_factor")
    ax.set_ylabel("h_b")
    ax.set_title(title)
    plt.colorbar(im,ax=ax)
    plt.tight_layout()
    plt.savefig(outpath,dpi=300)
    plt.close()

def main(csvfile):
    df=pd.read_csv(csvfile)
    ensure(OUTDIR)
    ensure(f"{OUTDIR}/summary")
    ensure(f"{OUTDIR}/reduced_datasets")
    ensure(f"{OUTDIR}/plots")

    bump_cols=[c for c in df.columns if c.endswith("_bumpangle")]
    if not bump_cols:
        raise RuntimeError("No *_bumpangle columns found.")

    df["average_bumpangle"]=df[bump_cols].mean(axis=1)
    df["abs_average_bumpangle"]=df["average_bumpangle"].abs()

    global_abs_max=df["abs_average_bumpangle"].max()
    global_avg_min=df["average_bumpangle"].min()
    global_avg_max=df["average_bumpangle"].max()

    ranking_rows=[]
    best_rows=[]

    grouped=df.groupby(["v","beta"],sort=True)

    for (v,beta),g in grouped:
        reduced=g[g["avg_zero_bump_timestep_percentage"]==0].copy()
        reduced=reduced[reduced["avg_one_bump_timestep_percentage"]>99.6].copy()
        reduced=reduced.sort_values("abs_average_bumpangle",ascending=True).reset_index(drop=True)

        safe_v=str(v).replace(".","p")
        safe_b=str(beta).replace(".","p")

        reduced.to_csv(
            f"{OUTDIR}/reduced_datasets/reduced_v{safe_v}_beta{safe_b}.csv",
            index=False
        )

        if reduced.empty:
            continue

        reduced.insert(0,"rank",np.arange(1,len(reduced)+1))
        reduced.to_csv(
            f"{OUTDIR}/summary/ranking_v{safe_v}_beta{safe_b}.csv",
            index=False
        )

        ranking_rows.append(reduced)

        best=reduced.iloc[0].copy()
        best_rows.append(best)

        plotdir=f"{OUTDIR}/plots/v{safe_v}_beta{safe_b}"
        ensure(plotdir)

        heatmap(
            reduced,
            "average_bumpangle",
            f"Average Bump Angle (v={v}, beta={beta})",
            f"{plotdir}/average_bumpangle.png",
            diverging=True,
            center=0.0,
            vmin=global_avg_min,
            vmax=global_avg_max,
        )

        heatmap(
            reduced,
            "abs_average_bumpangle",
            f"|Average Bump Angle| (v={v}, beta={beta})",
            f"{plotdir}/abs_average_bumpangle.png",
            diverging=False,
            vmin=0,
            vmax=global_abs_max,
        )

        heatmap(
            reduced,
            "avg_zero_bump_timestep_percentage",
            f"Zero Bump % (v={v}, beta={beta})",
            f"{plotdir}/avg_zero_bump_timestep_percentage.png",
        )

        heatmap(
            reduced,
            "avg_one_bump_timestep_percentage",
            f"One Bump % (v={v}, beta={beta})",
            f"{plotdir}/avg_one_bump_timestep_percentage.png",
        )

    if ranking_rows:
        pd.concat(ranking_rows,ignore_index=True).to_csv(
            f"{OUTDIR}/summary/ranking_per_v_beta.csv",
            index=False
        )

    if best_rows:
        pd.DataFrame(best_rows).to_csv(
            f"{OUTDIR}/summary/best_parameters_per_v_beta.csv",
            index=False
        )

    print("Analysis complete.")
    print("Results written to:",OUTDIR)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python parameter_sweep_analysis.py input.csv")
        sys.exit(1)
    main(sys.argv[1])