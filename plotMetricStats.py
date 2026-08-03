import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

csv_file = "./ztParameterStudy4/parameter_study_results.csv"      # <-- change this
output_dir = "./ztParameterStudy4/parameter_study_analysis"

os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv(csv_file)

# =============================================================================
# Reject invalid runs
# =============================================================================

df = df[df["avg_zero_bump_timestep_percentage"] == 0].copy()

print(f"Remaining parameter combinations: {len(df)}")

# =============================================================================
# Compute signed average bump angle
# =============================================================================

bumpangle_columns = [c for c in df.columns if c.endswith("_bumpangle")]

df["avg_bumpangle"] = df[bumpangle_columns].mean(axis=1)

# =============================================================================
# Ranking
# =============================================================================

df["angle_error"] = df["avg_bumpangle"].abs()
df["bump_error"] = (df["avg_number_of_bumps"] - 1).abs()

ranked = df.sort_values(
    ["angle_error", "bump_error"],
    ascending=[True, True]
).reset_index(drop=True)

# -------------------------------------------------------------------------
# Absolute rank (1 = best)
# -------------------------------------------------------------------------

ranked["rank"] = np.arange(1, len(ranked) + 1)

min_marker_size = 40
max_marker_size = 300

ranked["marker_size"] = np.interp(
    ranked["rank"],
    [1, len(ranked)],
    [max_marker_size, min_marker_size],
)

# Save ranked table
ranked.to_csv(
    os.path.join(output_dir, "ranked_parameter_combinations.csv"),
    index=False
)

print("\nTop 10 parameter combinations:\n")
print(
    ranked[
        [
            "normalizing_factor",
            "h_b",
            "v",
            "beta",
            "avg_bumpangle",
            "avg_number_of_bumps",
            "avg_total_active_neurons",
            "angle_error",
            "bump_error",
        ]
    ].head(10)
)

# =============================================================================
# Plotting
# =============================================================================

max_abs_angle = ranked["avg_bumpangle"].abs().max()

unique_pairs = (
    ranked[["v", "beta"]]
    .drop_duplicates()
    .sort_values(["v", "beta"])
)

for _, pair in unique_pairs.iterrows():

    v = pair["v"]
    beta = pair["beta"]

    subset = ranked[
        (ranked["v"] == v) &
        (ranked["beta"] == beta)
    ]
    
    
    best = subset.nsmallest(1, "rank")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        constrained_layout=True
    )

    # -------------------------------------------------------------------------
    # Average bump angle
    # -------------------------------------------------------------------------

    sc = axes[0].scatter(
        subset["normalizing_factor"],
        subset["h_b"],
        c=subset["avg_bumpangle"],
        cmap="RdBu_r",
        s=subset["marker_size"],
        edgecolor="black",
        vmin=-max_abs_angle,
        vmax=max_abs_angle,
    )

    axes[0].scatter(
        best["normalizing_factor"],
        best["h_b"],
        s=350,
        facecolors="none",
        edgecolors="gold",
        linewidths=2.5,
    )

    axes[0].set_title("Average Bump Angle")
    axes[0].set_xlabel("normalizing_factor")
    axes[0].set_ylabel("h_b")
    plt.colorbar(sc, ax=axes[0])

    # -------------------------------------------------------------------------
    # Average number of bumps
    # -------------------------------------------------------------------------

    sc = axes[1].scatter(
        subset["normalizing_factor"],
        subset["h_b"],
        c=subset["avg_number_of_bumps"],
        cmap="viridis",
        s=150,
        edgecolor="black",
    )

    axes[1].scatter(
        best["normalizing_factor"],
        best["h_b"],
        s=350,
        facecolors="none",
        edgecolors="gold",
        linewidths=2.5,
    )

    axes[1].set_title("Average Number of Bumps")
    axes[1].set_xlabel("normalizing_factor")
    axes[1].set_ylabel("h_b")
    plt.colorbar(sc, ax=axes[1])

    # -------------------------------------------------------------------------
    # Average active neurons
    # -------------------------------------------------------------------------

    sc = axes[2].scatter(
        subset["normalizing_factor"],
        subset["h_b"],
        c=subset["avg_total_active_neurons"],
        cmap="viridis",
        s=150,
        edgecolor="black",
    )

    axes[2].scatter(
        best["normalizing_factor"],
        best["h_b"],
        s=350,
        facecolors="none",
        edgecolors="gold",
        linewidths=2.5,
    )

    axes[2].set_title("Average Active Neurons")
    axes[2].set_xlabel("normalizing_factor")
    axes[2].set_ylabel("h_b")
    plt.colorbar(sc, ax=axes[2])

    fig.suptitle(
        f"Parameter Study (v={v}, beta={beta})",
        fontsize=16
    )

    plt.savefig(
        os.path.join(
            output_dir,
            f"parameter_map_v_{v}_beta_{beta}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

print(f"\nResults saved to: {output_dir}")