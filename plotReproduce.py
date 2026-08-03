import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = "./ztParameterStudySample3/spin_history2.csv"
OUTPUT_PDF = "./ztParameterStudySample3/spin_history_all_runs2.pdf"

SPINS_PER_RUN = 120
RUNS_PER_PAGE = 2

# Figure size in inches:  A4
FIGURE_SIZE = (8.27, 11.69)

# Angle represented by the 120 ring neurons
ANGLE_MIN = -180
ANGLE_MAX = 180


# =============================================================================
# LOAD SPIN HISTORY CSV
# =============================================================================

def load_spin_histories(csv_path):
    """
    Load spin_history.csv produced by the reproduction script.

    Expected format:

        run,seed,timestep_0,timestep_1,...,timestep_N

    Each timestep cell contains a 120-character binary string:

        001101010...

    Returns
    -------
    runs : list of dict

        Each dict contains:

            {
                "run": int,
                "seed": int,
                "spins_history": ndarray
            }

        spins_history shape:

            (n_timesteps, 120)
    """

    runs = []

    with open(
        csv_path,
        "r",
        newline=""
    ) as f:

        reader = csv.reader(f)

        header = next(reader)

        # First two columns:
        #
        # run
        # seed
        #
        # Everything after that is a timestep.
        timestep_columns = header[2:]

        n_timesteps = len(
            timestep_columns
        )

        print(
            f"Timesteps found: "
            f"{n_timesteps}"
        )

        for row in reader:

            if not row:
                continue

            run_number = int(
                row[0]
            )

            seed = int(
                row[1]
            )

            spin_cells = row[2:]

            if len(spin_cells) != n_timesteps:

                raise RuntimeError(
                    f"Run {run_number}: "
                    f"expected {n_timesteps} timesteps, "
                    f"found {len(spin_cells)}"
                )

            spins_history = np.zeros(
                (
                    n_timesteps,
                    SPINS_PER_RUN
                ),
                dtype=np.int8
            )

            for t, spin_string in enumerate(
                spin_cells
            ):

                spin_string = (
                    spin_string.strip()
                )

                if len(spin_string) != SPINS_PER_RUN:

                    raise RuntimeError(
                        f"Run {run_number}, "
                        f"timestep {t}: "
                        f"expected "
                        f"{SPINS_PER_RUN} spins, "
                        f"found "
                        f"{len(spin_string)}"
                    )

                spins_history[
                    t,
                    :
                ] = np.fromiter(
                    (
                        int(x)
                        for x in spin_string
                    ),
                    dtype=np.int8,
                    count=SPINS_PER_RUN
                )

            runs.append(
                {
                    "run": run_number,
                    "seed": seed,
                    "spins_history":
                        spins_history,
                }
            )

    return runs


# =============================================================================
# PLOT ONE SPIN HISTORY
# =============================================================================

def plot_spin_history(
    ax,
    spins_history,
    run_number,
    seed,
):
    """
    Plot one complete spin history.

    spins_history shape:

        timesteps x 120

    Transpose before imshow so:

        x-axis = time
        y-axis = ring neuron / angle
    """

    L = spins_history.shape[0]

    # Equivalent to your original plotting code:
    #
    # ax.imshow(
    #     spins_history.T,
    #     aspect='auto',
    #     origin='lower',
    #     cmap='gray',
    #     interpolation='nearest',
    #     extent=[0, L, -180, 180]
    # )

    ax.imshow(
        spins_history.T,
        aspect="auto",
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        extent=[
            0,
            L,
            ANGLE_MIN,
            ANGLE_MAX
        ],
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"Run {run_number}  |  "
        f"Seed {seed}",
        fontsize=10,
        pad=4
    )

    ax.set_ylabel(
        "angle (deg)",
        fontsize=9
    )

    ax.set_xlabel(
        "time step",
        fontsize=9
    )

    # Useful angular tick positions
    ax.set_yticks(
        [
            -180,
            -90,
            0,
            90,
            180
        ]
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.set_xlim(
        0,
        L
    )


# =============================================================================
# GENERATE MULTIPAGE PDF
# =============================================================================

def generate_pdf(
    runs,
    output_pdf
):

    if not runs:

        raise RuntimeError(
            "No spin histories found."
        )

    total_runs = len(
        runs
    )

    total_pages = (
        total_runs
        + RUNS_PER_PAGE
        - 1
    ) // RUNS_PER_PAGE

    print()
    print("=" * 70)
    print("GENERATING SPIN HISTORY PDF")
    print("=" * 70)

    print(
        f"Total runs       : "
        f"{total_runs}"
    )

    print(
        f"Runs per page    : "
        f"{RUNS_PER_PAGE}"
    )

    print(
        f"Total PDF pages  : "
        f"{total_pages}"
    )

    print(
        f"Output           : "
        f"{output_pdf}"
    )

    print()

    with PdfPages(
        output_pdf
    ) as pdf:

        for page_index in range(
            total_pages
        ):

            start_index = (
                page_index
                * RUNS_PER_PAGE
            )

            end_index = min(
                start_index
                + RUNS_PER_PAGE,
                total_runs
            )

            page_runs = runs[
                start_index:end_index
            ]

            # -------------------------------------------------------------
            # Create landscape page
            # -------------------------------------------------------------

            fig, axes = plt.subplots(
                RUNS_PER_PAGE,
                1,
                figsize=FIGURE_SIZE,
                squeeze=False
            )

            axes = axes[:, 0]

            # -------------------------------------------------------------
            # Plot up to 4 runs
            # -------------------------------------------------------------

            for plot_index in range(
                RUNS_PER_PAGE
            ):

                ax = axes[
                    plot_index
                ]

                if plot_index < len(
                    page_runs
                ):

                    run_data = page_runs[
                        plot_index
                    ]

                    plot_spin_history(
                        ax=ax,
                        spins_history=(
                            run_data[
                                "spins_history"
                            ]
                        ),
                        run_number=(
                            run_data[
                                "run"
                            ]
                        ),
                        seed=(
                            run_data[
                                "seed"
                            ]
                        ),
                    )

                else:

                    # Hide unused plots on final page.
                    ax.axis(
                        "off"
                    )

            # -------------------------------------------------------------
            # Page title
            # -------------------------------------------------------------

            first_run = page_runs[
                0
            ]["run"]

            last_run = page_runs[
                -1
            ]["run"]

            fig.suptitle(
                (
                    "Ring Attractor Spin History"
                    "\n"
                    f"Runs "
                    f"{first_run}–{last_run}"
                    f"  |  "
                    f"Page "
                    f"{page_index + 1}/"
                    f"{total_pages}"
                ),
                fontsize=13,
                y=0.985
            )

            # Adjust spacing.
            fig.subplots_adjust(
                left=0.10,
                right=0.97,
                bottom=0.06,
                top=0.93,
                hspace=0.45
            )

            # -------------------------------------------------------------
            # Save this page into PDF
            # -------------------------------------------------------------

            pdf.savefig(
                fig,
                orientation="portrait"
            )

            plt.close(
                fig
            )

            print(
                f"Saved page "
                f"{page_index + 1:02d}/"
                f"{total_pages} "
                f"| runs "
                f"{first_run}-"
                f"{last_run}",
                flush=True
            )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"PDF saved to: "
        f"{output_pdf}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print(
        f"Loading spin histories from:"
    )
    print(
        INPUT_CSV
    )

    runs = load_spin_histories(
        INPUT_CSV
    )

    print(
        f"Runs loaded: "
        f"{len(runs)}"
    )

    if runs:

        print(
            "Spin history shape per run: "
            f"{runs[0]['spins_history'].shape}"
        )

    generate_pdf(
        runs=runs,
        output_pdf=OUTPUT_PDF
    )


if __name__ == "__main__":
    main()