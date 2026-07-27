import csv
import math
import numpy as np


# =============================================================================
# HARDCODED CONFIGURATION
# =============================================================================

INPUT_CSV = "./data.csv"
SEEDS_FILE = "./ztParameterStudy/seeds.txt"
OUTPUT_CSV = "./ztParameterStudy/spin_history1.csv"

DB_THRESHOLD = 80.0

# Chosen parameter set from parameter study
NORMALIZING_FACTOR = 0.15   # <-- change
H_B = 0.01              # <-- change
V = 0.5                     # <-- change
BETA = 400                  # <-- change

NS = 120
UPDATES_PER_STEP = NS * 4

EXPECTED_N_SEEDS = 50

# NORMALIZING_FACTORS = [0.20, 0.15]

# H_B_VALUES = [0.10, 0.05]

# V_VALUES = [0.5]

# BETA_VALUES = [400]

# N_SEEDS = 3

# =============================================================================
# LOAD EXACT SAVED SEEDS
# =============================================================================

def load_seeds(path):
    with open(path, "r") as f:
        seeds = [
            int(line.strip())
            for line in f
            if line.strip()
        ]

    if len(seeds) != EXPECTED_N_SEEDS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_SEEDS} seeds, "
            f"but found {len(seeds)} in {path}"
        )

    return np.asarray(seeds, dtype=np.int64)


# =============================================================================
# INPUT LOADING AND NORMALIZATION
# Same logic as parameter-study script
# =============================================================================

def load_and_normalize_doa(input_path, db_threshold):

    raw_rows = []
    all_values = []
    timestamps = []

    with open(input_path, newline="") as f:

        reader = csv.reader(f)
        header = next(reader)

        robot_x_idx = header.index("robot_x")

        doa_headers = header[2:robot_x_idx]

        csv_angles = [
            int(float(angle))
            for angle in doa_headers
        ]

        for row in reader:

            db_spl = float(
                str(row[1])
                .replace("[", "")
                .replace("]", "")
                .strip()
            )

            # Same strict threshold as parameter study
            if db_spl <= db_threshold:
                continue

            values = np.asarray(
                [
                    float(x)
                    for x in row[2:robot_x_idx]
                ],
                dtype=np.float64
            )

            full_array = np.zeros(
                NS,
                dtype=np.float64
            )

            for angle, value in zip(
                csv_angles,
                values
            ):

                idx = (angle + 180) // 3

                if 0 <= idx < NS:

                    full_array[idx] = value

                    # Only real DOA measurements are used
                    # for global normalization range.
                    all_values.append(value)

            raw_rows.append(full_array)
            timestamps.append(row[0])

    if not raw_rows:
        raise RuntimeError(
            f"No rows found with dB_SPL > {db_threshold}"
        )

    all_values = np.asarray(
        all_values,
        dtype=np.float64
    )

    global_min = float(
        np.min(all_values)
    )

    global_max = float(
        np.max(all_values)
    )

    denom = global_max - global_min

    doa_base = np.zeros(
        (len(raw_rows), NS),
        dtype=np.float64
    )

    for t, arr in enumerate(raw_rows):

        nonzero = arr != 0.0

        if denom != 0.0:

            doa_base[t, nonzero] = (
                arr[nonzero] - global_min
            ) / denom

    return (
        np.ascontiguousarray(doa_base),
        timestamps
    )


# =============================================================================
# PRECOMPUTE INTERACTION KERNEL
# =============================================================================

def precompute_kernel(v):

    thetas = np.linspace(
        -np.pi,
        np.pi,
        NS,
        endpoint=False
    )

    kernel_matrix = np.zeros(
        (NS, NS),
        dtype=np.float64
    )

    for i in range(NS):

        alpha_i = thetas[i]

        angle_diffs = np.abs(
            (
                thetas
                - alpha_i
                + np.pi
            )
            % (2.0 * np.pi)
            - np.pi
        )

        kernel = np.cos(
            np.pi
            * (
                (
                    angle_diffs
                    / np.pi
                ) ** v
            )
        )

        # Same behavior as parameter-study code:
        # j_curr[i] = 0
        kernel[i] = 0.0

        kernel_matrix[i, :] = kernel

    return np.ascontiguousarray(
        kernel_matrix
    )


# =============================================================================
# RUN ONE SEED AND SAVE COMPLETE SPIN HISTORY
# =============================================================================

def run_single_seed(
    doa_base,
    kernel,
    normalizing_factor,
    h_b,
    beta,
    seed,
):

    # Must match parameter-study RNG exactly.
    rng = np.random.default_rng(
        int(seed)
    )

    n_timesteps = doa_base.shape[0]

    # Same initial spin generation as parameter study.
    spins = rng.integers(
        0,
        2,
        size=NS,
        dtype=np.int8
    )

    interaction_field = (
        kernel @ spins
    )

    # Shape:
    #
    #     timesteps x 120
    #
    spin_history = np.zeros(
        (n_timesteps, NS),
        dtype=np.int8
    )

    for t in range(n_timesteps):

        h_ext = (
            doa_base[t]
            * normalizing_factor
        )

        for _ in range(
            UPDATES_PER_STEP
        ):

            i = int(
                rng.integers(
                    0,
                    NS
                )
            )

            interaction_sum = (
                interaction_field[i]
            )

            effective_field = (
                interaction_sum
                / (NS - 1)
                + h_ext[i]
                - h_b
            )

            delta_H = (
                effective_field
                * (
                    2.0
                    * spins[i]
                    - 1.0
                )
            )

            should_flip = False

            if delta_H < 0.0:

                should_flip = True

            else:

                p = math.exp(
                    -beta
                    * delta_H
                )

                if rng.random() < p:

                    should_flip = True

            if should_flip:

                old_spin = int(
                    spins[i]
                )

                new_spin = (
                    1 - old_spin
                )

                delta_spin = (
                    new_spin
                    - old_spin
                )

                spins[i] = new_spin

                interaction_field += (
                    kernel[:, i]
                    * delta_spin
                )

        # IMPORTANT:
        # Save state AFTER all Monte Carlo updates
        # for this DOA timestep.
        spin_history[t, :] = spins

    return spin_history


# =============================================================================
# SAVE 50 x TIMESTEPS CSV
# =============================================================================

def save_spin_history_csv(
    output_path,
    seeds,
    all_histories,
):

    n_seeds = len(seeds)
    n_timesteps = all_histories.shape[1]

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        # First two columns identify each run.
        header = [
            "run",
            "seed",
        ]

        header.extend(
            [
                f"timestep_{t}"
                for t in range(n_timesteps)
            ]
        )

        writer.writerow(header)

        for run_idx in range(n_seeds):

            row = [
                run_idx + 1,
                int(seeds[run_idx]),
            ]

            for t in range(n_timesteps):

                spins = all_histories[
                    run_idx,
                    t,
                    :
                ]

                # Store the 120-neuron state as:
                #
                # 0100111010...
                #
                spin_string = "".join(
                    str(int(x))
                    for x in spins
                )

                row.append(
                    spin_string
                )

            writer.writerow(row)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("REPRODUCING SELECTED PARAMETER SET")
    print("=" * 70)

    print()
    print("Parameters:")
    print(
        f"  NORMALIZING_FACTOR = "
        f"{NORMALIZING_FACTOR}"
    )
    print(
        f"  H_B                = "
        f"{H_B}"
    )
    print(
        f"  V                  = "
        f"{V}"
    )
    print(
        f"  BETA               = "
        f"{BETA}"
    )
    print(
        f"  DB_THRESHOLD       = "
        f"{DB_THRESHOLD}"
    )

    # -------------------------------------------------------------------------
    # Load exact seeds from previous study
    # -------------------------------------------------------------------------

    seeds = load_seeds(
        SEEDS_FILE
    )

    print()
    print(
        f"Loaded {len(seeds)} seeds "
        f"from {SEEDS_FILE}"
    )

    # -------------------------------------------------------------------------
    # Load input exactly as parameter study
    # -------------------------------------------------------------------------

    doa_base, timestamps = (
        load_and_normalize_doa(
            INPUT_CSV,
            DB_THRESHOLD
        )
    )

    n_timesteps = (
        doa_base.shape[0]
    )

    print(
        f"Retained DOA timesteps: "
        f"{n_timesteps}"
    )

    # -------------------------------------------------------------------------
    # Kernel for selected v
    # -------------------------------------------------------------------------

    print(
        "Precomputing interaction kernel..."
    )

    kernel = precompute_kernel(
        V
    )

    # -------------------------------------------------------------------------
    # Allocate complete result
    #
    # Shape:
    #
    #     50 x timesteps x 120
    #
    # -------------------------------------------------------------------------

    all_histories = np.zeros(
        (
            len(seeds),
            n_timesteps,
            NS
        ),
        dtype=np.int8
    )

    # -------------------------------------------------------------------------
    # Reproduce all 50 runs
    # -------------------------------------------------------------------------

    for run_idx, seed in enumerate(
        seeds
    ):

        print(
            f"Run "
            f"{run_idx + 1:02d}/"
            f"{len(seeds)} "
            f"| seed={int(seed)}",
            flush=True
        )

        history = run_single_seed(
            doa_base=doa_base,
            kernel=kernel,
            normalizing_factor=NORMALIZING_FACTOR,
            h_b=H_B,
            beta=BETA,
            seed=int(seed),
        )

        all_histories[
            run_idx,
            :,
            :
        ] = history

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_spin_history_csv(
        OUTPUT_CSV,
        seeds,
        all_histories
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"Spin history saved to: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Runs: {len(seeds)}"
    )

    print(
        f"Timesteps per run: "
        f"{n_timesteps}"
    )

    print(
        f"Neurons per state: "
        f"{NS}"
    )

    print(
        "Internal history shape: "
        f"{all_histories.shape}"
    )

    print(
        "CSV data layout: "
        "one row per seed/run, "
        "one spin-state column per timestep"
    )


if __name__ == "__main__":
    main()