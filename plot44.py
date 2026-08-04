import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

METRICS = [
    (
        "avg_number_of_bumps",
        "Average Number of Bumps",
    ),
    (
        "avg_total_active_neurons",
        "Average Total Active Neurons",
    ),
    (
        "avg_bump_width",
        "Average Bump Width",
    ),
    (
        "avg_bumpangle",
        "Average Bump Angle",
    ),
    (
        "avg_zero_bump_timestep_percentage",
        "Average Zero-Bump Timesteps (%)",
    ),
    (
        "avg_one_bump_timestep_percentage",
        "Average One-Bump Timesteps (%)",
    ),
]

REQUIRED_COLUMNS = [
    "normalizing_factor",
    "h_b",
    "v",
    "beta",
    "avg_number_of_bumps",
    "avg_total_active_neurons",
    "avg_bump_width",
    "avg_bumpangle",
    "avg_zero_bump_timestep_percentage",
    "avg_one_bump_timestep_percentage",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def set_sparse_ticks(ax, x_values, y_values, max_ticks=8):
    """
    Show at most `max_ticks` tick labels per axis.
    """

    def make_ticks(values):
        n = len(values)

        if n <= max_ticks:
            indices = np.arange(n)
        else:
            indices = np.linspace(
                0,
                n - 1,
                max_ticks,
                dtype=int,
            )
            indices = np.unique(indices)

        labels = [f"{values[i]:g}" for i in indices]
        return indices, labels

    x_idx, x_labels = make_ticks(x_values)
    y_idx, y_labels = make_ticks(y_values)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(
        x_labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(y_idx)
    ax.set_yticklabels(y_labels)

def format_parameter_for_filename(value):
    """
    Convert a parameter value into a clean, filename-safe string.

    Examples:
        0.3   -> "0.3"
        50.0  -> "50"
        1.25  -> "1.25"
    """
    if pd.isna(value):
        return "nan"

    try:
        value = float(value)

        if value.is_integer():
            return str(int(value))

        return f"{value:g}"

    except (TypeError, ValueError):
        return str(value).replace(" ", "_")


def calculate_global_color_scales(df):
    """
    Calculate one global color scale per metric using the entire CSV.

    Returns
    -------
    dict
        {
            metric_name: (global_min, global_max),
            ...
        }
    """
    global_scales = {}

    print()
    print("=" * 70)
    print("GLOBAL COLOR SCALES")
    print("=" * 70)

    for metric, title in METRICS:

        values = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.empty:
            raise ValueError(
                f"Metric '{metric}' contains no finite values."
            )

        global_min = finite_values.min()
        global_max = finite_values.max()

        global_scales[metric] = (
            global_min,
            global_max,
        )

        print(
            f"{title:<35} "
            f"min = {global_min:g}, "
            f"max = {global_max:g}"
        )

    return global_scales


# =============================================================================
# PLOT ONE (v, beta) PAIR
# =============================================================================

def plot_single_parameter_pair(
    subset,
    v_value,
    beta_value,
    global_scales,
    output_path,
):
    """
    Create the four heatmaps for one (v, beta) pair.

    X-axis:
        normalizing_factor

    Y-axis:
        h_b

    Color scales:
        Fixed globally for each metric across the entire CSV.
    """

    n_metrics = len(METRICS)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(24, 12),
        constrained_layout=True,
    )

    axes = axes.flatten()

    axes = np.atleast_1d(axes).flatten()

    for ax, (metric, title) in zip(
        axes,
        METRICS,
    ):

        # ---------------------------------------------------------------------
        # Create heatmap matrix
        #
        # rows    = h_b
        # columns = normalizing_factor
        #
        # pivot_table is used instead of pivot so duplicate combinations
        # do not crash the script.
        #
        # If duplicates exist, their metric values are averaged.
        # ---------------------------------------------------------------------

        heatmap = subset.pivot_table(
            index="h_b",
            columns="normalizing_factor",
            values=metric,
            aggfunc="mean",
        )

        # Sort both axes numerically ascending.
        heatmap = heatmap.sort_index(
            ascending=True
        )

        heatmap = heatmap.sort_index(
            axis=1,
            ascending=True,
        )

        x_values = heatmap.columns.to_numpy()
        y_values = heatmap.index.to_numpy()

        data = heatmap.to_numpy(
            dtype=float
        )

        # ---------------------------------------------------------------------
        # Global color scale for this metric
        # ---------------------------------------------------------------------

        global_min, global_max = (
            global_scales[metric]
        )

        # If all values for a metric are identical, Matplotlib needs
        # a non-zero color range.
        if np.isclose(
            global_min,
            global_max,
        ):
            padding = (
                abs(global_min) * 0.01
                if global_min != 0
                else 0.01
            )

            vmin = global_min - padding
            vmax = global_max + padding

        else:
            vmin = global_min
            vmax = global_max

        # ---------------------------------------------------------------------
        # Heatmap
        # ---------------------------------------------------------------------

        # Use an inverted colormap for the zero-bump percentage metric
        cmap = "viridis_r" if metric == "avg_zero_bump_timestep_percentage" else "viridis"

        image = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # ---------------------------------------------------------------------
        # Colorbar
        # ---------------------------------------------------------------------

        cbar = fig.colorbar(
            image,
            ax=ax,
        )

        cbar.set_label(
            title,
            fontsize=11,
        )

        set_sparse_ticks(
            ax,
            x_values,
            y_values,
            max_ticks=8,   # change to 6, 10, etc.
        )    
        # ---------------------------------------------------------------------
        # Labels
        # ---------------------------------------------------------------------

        ax.set_xlabel(
            "NORMALIZING_FACTOR",
            fontsize=12,
        )

        ax.set_ylabel(
            "h_b",
            fontsize=12,
        )

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
        )
        
    # -------------------------------------------------------------------------
    # Plot 5: Exact zero-bump parameter combinations
    # -------------------------------------------------------------------------

    ax = axes[5]

    zero_map = subset.pivot_table(
        index="h_b",
        columns="normalizing_factor",
        values="avg_zero_bump_timestep_percentage",
        aggfunc="mean",
    )

    zero_map = zero_map.sort_index().sort_index(axis=1)

    x_values = zero_map.columns.to_numpy()
    y_values = zero_map.index.to_numpy()

    # Plot only exact zeros
    mask = np.isclose(zero_map.to_numpy(dtype=float), 0.0)

    rows, cols = np.where(mask)

    # Blank background
    ax.imshow(
        np.zeros_like(mask, dtype=float),
        origin="lower",
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    # Red markers where zero-bump percentage is exactly zero
    ax.scatter(
        cols,
        rows,
        color="red",
        s=120,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
    )

    set_sparse_ticks(ax, x_values, y_values)

    ax.set_xlabel("NORMALIZING_FACTOR")
    ax.set_ylabel("h_b")
    ax.set_title(
        "Exact 0% Zero-Bump Timesteps",
        fontsize=14,
        fontweight="bold",
    )
    
    # -------------------------------------------------------------------------
    # Plot 8: Parameter combinations with >=99% one-bump timesteps
    # -------------------------------------------------------------------------

    ax = axes[7]

    one_map = subset.pivot_table(
        index="h_b",
        columns="normalizing_factor",
        values="avg_one_bump_timestep_percentage",
        aggfunc="mean",
    )

    one_map = one_map.sort_index().sort_index(axis=1)

    x_values = one_map.columns.to_numpy()
    y_values = one_map.index.to_numpy()

    mask = one_map.to_numpy(dtype=float) >= 99.0

    rows, cols = np.where(mask)

    ax.imshow(
        np.zeros_like(mask, dtype=float),
        origin="lower",
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    ax.scatter(
        cols,
        rows,
        color="red",
        s=120,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
    )

    set_sparse_ticks(ax, x_values, y_values)

    ax.set_xlabel("NORMALIZING_FACTOR")
    ax.set_ylabel("h_b")
    ax.set_title(
        "≥99% One-Bump Timesteps",
        fontsize=14,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Main title
    # -------------------------------------------------------------------------

    fig.suptitle(
        (
            "Ring Attractor Parameter Study\n"
            f"v = {v_value:g}, "
            f"beta = {beta_value:g}"
        ),
        fontsize=18,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Save PDF
    # -------------------------------------------------------------------------

    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
    )

    # Important when generating many plots:
    # release figure memory instead of calling plt.show().
    plt.close(fig)


# =============================================================================
# MAIN PARAMETER STUDY PLOTTING FUNCTION
# =============================================================================

def plot_all_parameter_pairs(
    csv_path,
    output_dir,
):
    """
    Generate one PDF for every existing (v, beta) pair in the CSV.

    Each PDF contains four heatmaps over:

        h_b x normalizing_factor

    Each metric uses a global color scale calculated from the entire CSV.
    """

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist:\n{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )
    
    # Average bump angle over all recorded timesteps
    bumpangle_columns = [
        c for c in df.columns
        if c.startswith("time") and c.endswith("_bumpangle")
    ]

    if not bumpangle_columns:
        raise ValueError("No time*_bumpangle columns found in CSV.")

    df["avg_bumpangle"] = (
        df[bumpangle_columns]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )

    # -------------------------------------------------------------------------
    # Validate required columns
    # -------------------------------------------------------------------------

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    if df.empty:
        raise ValueError(
            "The input CSV contains no rows."
        )

    # -------------------------------------------------------------------------
    # Convert parameter/metric columns to numeric
    # -------------------------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Remove rows without valid parameter identifiers
    # -------------------------------------------------------------------------

    invalid_parameter_rows = df[
        [
            "normalizing_factor",
            "h_b",
            "v",
            "beta",
        ]
    ].isna().any(
        axis=1
    )

    n_invalid = invalid_parameter_rows.sum()

    if n_invalid > 0:

        print()
        print(
            f"[WARNING] Removing {n_invalid} rows "
            "with invalid parameter values."
        )

        df = df.loc[
            ~invalid_parameter_rows
        ].copy()

    if df.empty:
        raise ValueError(
            "No valid parameter rows remain "
            "after numeric conversion."
        )

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Dataset summary
    # -------------------------------------------------------------------------

    unique_nf = np.sort(
        df[
            "normalizing_factor"
        ].unique()
    )

    unique_hb = np.sort(
        df[
            "h_b"
        ].unique()
    )

    unique_v = np.sort(
        df[
            "v"
        ].unique()
    )

    unique_beta = np.sort(
        df[
            "beta"
        ].unique()
    )

    print("=" * 70)
    print("PARAMETER STUDY PDF GENERATOR")
    print("=" * 70)

    print(
        f"Input CSV          : {csv_path}"
    )

    print(
        f"Output directory   : {output_dir}"
    )

    print(
        f"Total CSV rows     : {len(df)}"
    )

    print(
        f"Normalizing factors: {len(unique_nf)}"
    )

    print(
        f"h_b values         : {len(unique_hb)}"
    )

    print(
        f"v values           : {len(unique_v)}"
    )

    print(
        f"beta values        : {len(unique_beta)}"
    )

    # -------------------------------------------------------------------------
    # Discover actual (v, beta) pairs from CSV
    #
    # We do NOT assume that the Cartesian product of unique v and beta
    # necessarily exists.
    # -------------------------------------------------------------------------

    parameter_pairs = (
        df[
            [
                "v",
                "beta",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "v",
                "beta",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Existing (v, beta) pairs: "
        f"{len(parameter_pairs)}"
    )

    # -------------------------------------------------------------------------
    # Calculate GLOBAL scales before generating any plots
    # -------------------------------------------------------------------------

    global_scales = (
        calculate_global_color_scales(
            df
        )
    )

    # -------------------------------------------------------------------------
    # Check duplicates
    # -------------------------------------------------------------------------

    combination_columns = [
        "normalizing_factor",
        "h_b",
        "v",
        "beta",
    ]

    duplicate_mask = df.duplicated(
        subset=combination_columns,
        keep=False,
    )

    duplicate_count = (
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        print()
        print(
            "[WARNING] "
            f"{duplicate_count} rows belong to duplicated "
            "(normalizing_factor, h_b, v, beta) combinations."
        )

        print(
            "Duplicate metric values will be averaged "
            "within each heatmap cell."
        )

    # -------------------------------------------------------------------------
    # Generate one PDF per existing (v, beta) pair
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING PDF FILES")
    print("=" * 70)

    generated_files = []

    total_pairs = len(
        parameter_pairs
    )

    for index, row in parameter_pairs.iterrows():

        v_value = row["v"]
        beta_value = row["beta"]

        # ---------------------------------------------------------------------
        # Select this exact parameter pair
        # ---------------------------------------------------------------------

        subset = df[
            (df["v"] == v_value)
            &
            (df["beta"] == beta_value)
        ].copy()

        if subset.empty:

            # This should not happen because pairs were discovered
            # directly from the CSV, but keep the check for robustness.
            print(
                f"[WARNING] Empty subset for "
                f"v={v_value:g}, "
                f"beta={beta_value:g}. "
                "Skipping."
            )

            continue

        # ---------------------------------------------------------------------
        # Filename
        # ---------------------------------------------------------------------

        v_string = (
            format_parameter_for_filename(
                v_value
            )
        )

        beta_string = (
            format_parameter_for_filename(
                beta_value
            )
        )

        filename = (
            f"parameter_study_"
            f"v_{v_string}_"
            f"beta_{beta_string}.pdf"
        )

        output_path = (
            output_dir
            / filename
        )

        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        print(
            f"[{index + 1}/{total_pairs}] "
            f"v = {v_value:g}, "
            f"beta = {beta_value:g}"
        )

        print(
            f"    rows : {len(subset)}"
        )

        print(
            f"    file : {filename}"
        )

        # ---------------------------------------------------------------------
        # Plot
        # ---------------------------------------------------------------------

        plot_single_parameter_pair(
            subset=subset,
            v_value=v_value,
            beta_value=beta_value,
            global_scales=global_scales,
            output_path=output_path,
        )

        generated_files.append(
            output_path
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"Generated PDFs : "
        f"{len(generated_files)}"
    )

    print(
        f"Saved in       : "
        f"{output_dir.resolve()}"
    )


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate one 4-panel parameter-study PDF "
            "for every existing (v, beta) pair in a CSV, "
            "using global per-metric color scales."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to parameter_study_results.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="parameter_study_plots",
        help=(
            "Directory where PDF files will be saved. "
            "Default: parameter_study_plots"
        ),
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    args = parse_args()

    plot_all_parameter_pairs(
        csv_path=args.input,
        output_dir=args.output_dir,
    )
    
    
    # python3 plot44.py --input ./ztParameterStudy/parameter_study_results.csv --output-dir ./ztParameterStudy/parameter_study_pdfs
        # python3 plot44.py --input ./ztParameterStudy2/parameter_study_results.csv --output-dir ./ztParameterStudy2/parameter_study_pdfs
            # python3 plot44.py --input ./ztParameterStudySample/parameter_study_results.csv --output-dir ./ztParameterStudySample/parameter_study_pdfs
                # python3 plot44.py --input ./ztLogParameterStudy/parameter_study_results.csv --output-dir ./ztLogParameterStudy/parameter_study_pdfs