import os
import csv
import json
import math
import argparse
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# python3 main.py \
#     --input ../ztLabCollection/ztProcessData/expAA1.csv \
#     --output-dir ./ztParameterStudy \
#     --db-threshold 80 \
#     --workers 8


NS = 120
UPDATES_PER_STEP = NS * 4

# # Full parameter sweep
# NORMALIZING_FACTORS = np.arange(0.5000, 0.0000, -0.05).tolist()

# # 0.1546

# H_B_VALUES = np.arange(0.5000, 0.0000, -0.05).tolist()

# Full parameter sweep
NORMALIZING_FACTORS = np.arange(0.5000, 0.0000, -0.005).tolist()

# 0.1546

H_B_VALUES = np.arange(0.2000, 0.0000, -0.005).tolist()

# 0.0122

V_VALUES = [0.5]

BETA_VALUES = [400]

# Small parameter sweep for testing

# NORMALIZING_FACTORS = [0.1546]

# H_B_VALUES = [0.0122]

# V_VALUES = [0.5]

# BETA_VALUES = [400]

# N_SEEDS = 3

# # Full parameter sweep
# NORMALIZING_FACTORS = np.round(
#     np.arange(1.00, 0.049, -0.05), 2
# ).tolist()

# H_B_VALUES = np.round(
#     np.arange(0.100, -0.0001, -0.005), 3
# ).tolist()

# V_VALUES = [0.3, 0.4, 0.5, 0.6]

# BETA_VALUES = [
#     100, 200, 300, 400, 500,
#     600, 700, 800, 900, 1000
# ]

# Number of stochastic repetitions per parameter combination
N_SEEDS = 50

# Used only to generate the fixed set of 50 seeds
MASTER_SEED = 100

# Gaps of 1 or 2 zeros are tolerated inside a bump.
# Gaps of >= 3 zeros separate bumps.
ZERO_GAP_TOLERANCE = 2


# =============================================================================
# GLOBAL WORKER DATA
# =============================================================================
#
# These are initialized once inside each multiprocessing worker.
# This avoids passing the DOA matrix and kernels with every task.
# =============================================================================

WORKER_DOA_BASE = None
WORKER_KERNELS = None
WORKER_SEEDS = None


def init_worker(doa_base, kernels, seeds):
    global WORKER_DOA_BASE
    global WORKER_KERNELS
    global WORKER_SEEDS

    WORKER_DOA_BASE = doa_base
    WORKER_KERNELS = kernels
    WORKER_SEEDS = seeds


# =============================================================================
# INPUT LOADING AND NORMALIZATION
# =============================================================================

def load_and_normalize_doa(input_path, db_threshold):
    """
    Load one DOA collection CSV.

    Only rows satisfying:

        dB_SPL > db_threshold

    are retained.

    DOA values are mapped into the 120-neuron ring exactly as in the
    original implementation.

    Normalization:

        (value - global_min) / (global_max - global_min)

    global_min/global_max are calculated only from actual DOA measurements
    in qualifying rows.

    Artificial zeros in the unused part of the 360-degree ring are NOT
    included when calculating min/max.

    NORMALIZING_FACTOR is NOT applied here. We create one normalized base
    matrix and apply each factor later during the parameter sweep.
    """

    raw_rows = []
    all_values = []
    timestamps = []
    db_values = []

    with open(input_path, newline="") as f:

        reader = csv.reader(f)
        header = next(reader)

        # robot_x_idx = header.index("robot_x")

        # DOA columns:
        # header[2] ... column immediately before robot_x
        doa_headers = header[2:]

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

            # Strictly greater than threshold
            if db_spl <= db_threshold:
                continue

            values = np.asarray(
                [
                    float(x)
                    for x in row[2:]
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

                    # Only actual DOA values participate
                    # in global min/max.
                    all_values.append(value)

            raw_rows.append(full_array)

            timestamps.append(row[0])
            db_values.append(db_spl)

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

        # Preserve original behavior:
        #
        # if v == 0:
        #     normalized value = 0
        #
        # Therefore artificial zero-filled ring positions
        # remain exactly zero.

        nonzero = arr != 0.0

        if denom != 0.0:

            doa_base[t, nonzero] = (
                arr[nonzero] - global_min
            ) / denom

    print()
    print("=" * 70)
    print("INPUT SUMMARY")
    print("=" * 70)

    print(f"Input file             : {input_path}")
    print(f"dB filter              : dB_SPL > {db_threshold}")
    print(f"Retained DOA timesteps : {len(raw_rows)}")

    print(
        f"Raw DOA value range    : "
        f"[{global_min:.6f}, {global_max:.6f}]"
    )

    return (
        np.ascontiguousarray(doa_base),
        timestamps,
        np.asarray(
            db_values,
            dtype=np.float64
        )
    )


# =============================================================================
# PRECOMPUTE INTERACTION KERNELS
# =============================================================================

def precompute_kernels():
    """
    Precompute the full 120 x 120 interaction matrix for each v.

    Original compute_J():

        alpha_i = ring.thetas[i]

        angle_diffs = abs(
            (ring.thetas - alpha_i + pi)
            % (2*pi)
            - pi
        )

        kernel = cos(
            pi * ((angle_diffs/pi) ** v)
        )

    Original compute_delta_H then does:

        j_curr[i] = 0

    so the diagonal is explicitly set to zero here.

    Output shape:

        (4, 120, 120)

    corresponding to:

        v = 0.3
        v = 0.4
        v = 0.5
        v = 0.6
    """

    thetas = np.linspace(
        -np.pi,
        np.pi,
        NS,
        endpoint=False
    )

    kernels = np.zeros(
        (
            len(V_VALUES),
            NS,
            NS
        ),
        dtype=np.float64
    )

    for v_idx, v in enumerate(V_VALUES):

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

            # Same as:
            #
            # j_curr[i] = 0
            #
            # in your original compute_delta_H()
            kernel[i] = 0.0

            kernels[
                v_idx,
                i,
                :
            ] = kernel

    return np.ascontiguousarray(
        kernels
    )


# =============================================================================
# FIXED RANDOM SEEDS
# =============================================================================

def generate_fixed_seeds(
    n_seeds=N_SEEDS,
    master_seed=MASTER_SEED
):
    """
    Generate 50 random but fixed seed values.

    The exact same 50 seeds are reused for EVERY parameter combination.
    """

    rng = np.random.default_rng(
        master_seed
    )

    seeds = rng.choice(
        np.arange(
            1,
            2_000_000_000,
            dtype=np.int64
        ),
        size=n_seeds,
        replace=False
    )

    return seeds.astype(
        np.int64
    )


def save_seeds(
    seeds,
    output_dir
):

    path = os.path.join(
        output_dir,
        "seeds.txt"
    )

    with open(path, "w") as f:

        for seed in seeds:

            f.write(
                f"{int(seed)}\n"
            )

    return path


# =============================================================================
# BUMP DETECTION
# =============================================================================

def calculate_bump_metrics(
    spins,
    zero_gap_tolerance=ZERO_GAP_TOLERANCE
):
    """
    Calculate bump metrics on a circular binary ring.

    Definition
    ----------

    Example:

        11110111

    One zero separates the active regions, therefore:

        1 bump
        width = 7 active neurons

    Example:

        11110011

    Two zeros are tolerated:

        1 bump
        width = 6 active neurons

    Example:

        1111000111

    Three zeros separate the regions:

        2 bumps

    The ring is circular, so neuron 119 connects to neuron 0.

    Width counts ACTIVE NEURONS ONLY.

    Tolerated zero gaps do not contribute to width.

    Returns
    -------

    n_bumps
    total_active
    list_of_bump_widths
    largest_bump_width
    """

    spins = np.asarray(
        spins,
        dtype=np.int8
    )

    total_active = int(
        np.sum(spins)
    )

    # Completely inactive ring
    if total_active == 0:

        return (
            0,
            0,
            [],
            0
        )

    # -------------------------------------------------------------------------
    # Find one active neuron.
    #
    # Starting traversal from an active neuron makes circular zero-run
    # detection straightforward.
    # -------------------------------------------------------------------------

    active_indices = np.flatnonzero(
        spins
    )

    first_active = int(
        active_indices[0]
    )

    # -------------------------------------------------------------------------
    # Detect long zero gaps.
    #
    # A zero run longer than the tolerance separates bumps.
    #
    # tolerance = 2:
    #
    #   gap 0 -> same bump
    #   gap 1 -> same bump
    #   gap 2 -> same bump
    #   gap 3+ -> new bump
    # -------------------------------------------------------------------------

    long_gaps = []

    zero_start = None
    zero_length = 0

    # Traverse exactly one full circular revolution.
    #
    # We begin after first_active and eventually return to it.
    for step in range(
        1,
        NS + 1
    ):

        idx = (
            first_active + step
        ) % NS

        if spins[idx] == 0:

            if zero_length == 0:

                zero_start = idx

            zero_length += 1

        else:

            if (
                zero_length
                > zero_gap_tolerance
            ):

                long_gaps.append(
                    (
                        zero_start,
                        zero_length
                    )
                )

            zero_start = None
            zero_length = 0

    # -------------------------------------------------------------------------
    # No separating gap means every active neuron belongs to one bump.
    # -------------------------------------------------------------------------

    if not long_gaps:

        return (
            1,
            total_active,
            [total_active],
            total_active
        )

    # Each long zero gap separates one circular bump.
    n_bumps = len(
        long_gaps
    )

    bump_widths = []

    # -------------------------------------------------------------------------
    # A bump is the circular region between two consecutive long gaps.
    #
    # Small zero gaps may occur inside this region, but width counts only 1s.
    # -------------------------------------------------------------------------

    for gap_index in range(
        n_bumps
    ):

        current_start, current_length = (
            long_gaps[gap_index]
        )

        next_start, _ = (
            long_gaps[
                (gap_index + 1)
                % n_bumps
            ]
        )

        # Begin immediately after the current separating gap.
        idx = (
            current_start
            + current_length
        ) % NS

        width = 0

        while idx != next_start:

            if spins[idx] == 1:

                width += 1

            idx = (
                idx + 1
            ) % NS

        bump_widths.append(
            width
        )

    largest_width = max(
        bump_widths
    )

    return (
        n_bumps,
        total_active,
        bump_widths,
        largest_width
    )


def calculate_bump_angle(spins):
    """
    Circular mean of all active neurons.

    Returns
    -------
    Angle in degrees in [-180, 180].

    If there are no active neurons,
    returns 0.0.
    """

    active = np.flatnonzero(spins)

    if active.size == 0:
        return 0.0

    thetas = np.linspace(
        -np.pi,
        np.pi,
        NS,
        endpoint=False
    )

    angle = np.angle(
        np.sum(
            np.exp(1j * thetas[active])
        )
    )

    return np.degrees(angle)

# =============================================================================
# ONE COMPLETE SIMULATION RUN
# =============================================================================

def run_single_seed(
    doa_base,
    kernel,
    normalizing_factor,
    h_b,
    beta,
    seed,
):
    """
    Run one complete stochastic ring simulation.

    IMPORTANT:

    One fresh ring is created for this seed.

    The SAME ring then evolves continuously through:

        DOA 1
        DOA 2
        DOA 3
        ...
        final DOA

    It is NOT reset between DOA rows.

    Optimization
    ------------

    Original code recalculates:

        dot(J[i], spins)

    for every Monte Carlo proposal.

    Instead, we maintain:

        interaction_field = J @ spins

    If one spin changes:

        old_spin -> new_spin

    then:

        delta_spin = new_spin - old_spin

    and every interaction sum can be updated exactly:

        interaction_field += J[:, i] * delta_spin

    This preserves the interaction mathematics while avoiding one
    full dot product for every proposal.
    """

    # Independent deterministic RNG for this simulation run.
    rng = np.random.default_rng(
        int(seed)
    )

    n_timesteps = (
        doa_base.shape[0]
    )
    
    bump_angles = np.zeros(
        n_timesteps,
        dtype=np.float64
    )

    # Equivalent distribution to:
    #
    # np.random.choice([1, 0], size=120)
    #
    # 50/50 binary initialization.
    spins = rng.integers(
        0,
        2,
        size=NS,
        dtype=np.int8
    )

    # -------------------------------------------------------------------------
    # Precompute interaction field for current state.
    #
    # interaction_field[i]
    #
    # is exactly:
    #
    # dot(kernel[i], spins)
    #
    # -------------------------------------------------------------------------

    interaction_field = (
        kernel @ spins
    )

    # Aggregation variables
    total_bump_count = 0

    total_active_neurons = 0

    total_all_bump_widths = 0

    total_number_of_detected_bumps = 0

    total_largest_bump_width = 0
    
    total_zero_bump_timesteps = 0

    # =========================================================================
    # Process complete DOA sequence
    # =========================================================================

    for t in range(
        n_timesteps
    ):

        # ---------------------------------------------------------------------
        # External input for this DOA timestep
        # ---------------------------------------------------------------------

        h_ext = (
            doa_base[t]
            * normalizing_factor
        )

        # ---------------------------------------------------------------------
        # Monte Carlo updates
        # ---------------------------------------------------------------------

        for _ in range(
            UPDATES_PER_STEP
        ):

            i = int(
                rng.integers(
                    0,
                    NS
                )
            )

            # This is exactly:
            #
            # interaction_sum =
            #     np.dot(kernel[i], spins)
            #
            # but already maintained incrementally.
            interaction_sum = (
                interaction_field[i]
            )

            # ================================================================
            # Original compute_delta_H mathematics
            # ================================================================
            #
            # Original:
            #
            # spin_flip = [
            #     spins[i],
            #     1 - spins[i]
            # ]
            #
            # interaction_energy =
            #     (
            #         interaction_sum
            #         * spin_flip
            #     )
            #     / (NS - 1)
            #
            # input_energy =
            #     h_ext[i]
            #     * spin_flip
            #
            # leak_energy =
            #     h_b
            #     * spin_flip
            #
            # H = -1 * (
            #     interaction_energy
            #     + input_energy
            #     - leak_energy
            # )
            #
            # delta_H =
            #     H[1] - H[0]
            #
            #
            # Algebraically:
            #
            # field =
            #
            #     interaction_sum/(NS-1)
            #     + h_ext[i]
            #     - h_b
            #
            #
            # delta_H =
            #
            #     field * (2*spin - 1)
            #
            # ================================================================

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

            # -----------------------------------------------------------------
            # Apply flip + incrementally update interaction field
            # -----------------------------------------------------------------

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

                spins[i] = (
                    new_spin
                )

                # Exact update:
                #
                # New interaction:
                #
                # J @ new_spins
                #
                # =
                #
                # J @ old_spins
                # +
                # J[:, i] * delta_spin
                #
                interaction_field += (
                    kernel[:, i]
                    * delta_spin
                )

        # ---------------------------------------------------------------------
        # Metrics after this DOA timestep
        # ---------------------------------------------------------------------

        (
            n_bumps,
            total_active,
            bump_widths,
            largest_width
        ) = calculate_bump_metrics(
            spins
        )
        
        bump_angles[t] = calculate_bump_angle(spins)
        
        if n_bumps == 0:
            total_zero_bump_timesteps += 1  

        total_bump_count += (
            n_bumps
        )

        total_active_neurons += (
            total_active
        )

        total_largest_bump_width += (
            largest_width
        )

        # Zero-bump states do NOT contribute a fake width=0 bump.
        if n_bumps > 0:

            total_all_bump_widths += (
                sum(bump_widths)
            )

            total_number_of_detected_bumps += (
                n_bumps
            )

    return (
        total_bump_count,
        total_active_neurons,
        total_all_bump_widths,
        total_number_of_detected_bumps,
        total_largest_bump_width,
        total_zero_bump_timesteps,
        bump_angles,
    )


# =============================================================================
# ONE PARAMETER COMBINATION × ALL 50 SEEDS
# =============================================================================

def run_parameter_combination(
    task
):
    """
    Run one parameter combination across all 50 fixed seeds.
    """

    (
        normalizing_factor,
        h_b,
        v,
        beta
    ) = task

    doa_base = (
        WORKER_DOA_BASE
    )

    kernels = (
        WORKER_KERNELS
    )

    seeds = (
        WORKER_SEEDS
    )

    # Map v to precomputed kernel.
    v_idx = V_VALUES.index(
        v
    )

    kernel = kernels[
        v_idx
    ]

    n_timesteps = (
        doa_base.shape[0]
    )

    n_seeds = len(
        seeds
    )

    # -------------------------------------------------------------------------
    # Aggregate across all 50 complete runs
    # -------------------------------------------------------------------------

    grand_bump_count = 0

    grand_active_neurons = 0

    grand_all_bump_widths = 0

    grand_number_of_detected_bumps = 0

    grand_largest_bump_width = 0
    
    grand_zero_bump_timesteps = 0
    
    grand_bump_angles = np.zeros(
        n_timesteps,
        dtype=np.float64
    )

    for seed in seeds:

        (
            total_bump_count,
            total_active_neurons,
            total_all_bump_widths,
            total_number_of_detected_bumps,
            total_largest_bump_width,
            total_zero_bump_timesteps,
            bump_angles,
        ) = run_single_seed(
            doa_base=doa_base,
            kernel=kernel,
            normalizing_factor=normalizing_factor,
            h_b=h_b,
            beta=beta,
            seed=int(seed),
        )
        
        grand_bump_angles += bump_angles

        grand_bump_count += (
            total_bump_count
        )

        grand_active_neurons += (
            total_active_neurons
        )

        grand_all_bump_widths += (
            total_all_bump_widths
        )

        grand_number_of_detected_bumps += (
            total_number_of_detected_bumps
        )

        grand_largest_bump_width += (
            total_largest_bump_width
        )
        
        grand_zero_bump_timesteps += total_zero_bump_timesteps

    # -------------------------------------------------------------------------
    # Average state-level metrics
    #
    # Number of ring states:
    #
    #     number of DOAs × 50 seeds
    # -------------------------------------------------------------------------

    total_states = (
        n_timesteps
        * n_seeds
    )

    avg_number_of_bumps = (
        grand_bump_count
        / total_states
    )

    avg_total_active_neurons = (
        grand_active_neurons
        / total_states
    )

    avg_largest_bump_width = (
        grand_largest_bump_width
        / total_states
    )
    
    avg_zero_bump_timestep_percentage = (
        100.0 * grand_zero_bump_timesteps / total_states
    )
    
    avg_bump_angles = grand_bump_angles / n_seeds

    # -------------------------------------------------------------------------
    # Average individual bump width
    #
    # Denominator is ACTUAL detected bumps, not ring states.
    # -------------------------------------------------------------------------

    if (
        grand_number_of_detected_bumps
        > 0
    ):

        avg_bump_width = (
            grand_all_bump_widths
            / grand_number_of_detected_bumps
        )

    else:

        avg_bump_width = (
            float("nan")
        )

    return (
        normalizing_factor,
        h_b,
        v,
        beta,
        avg_number_of_bumps,
        avg_total_active_neurons,
        avg_bump_width,
        avg_largest_bump_width,
        avg_zero_bump_timestep_percentage,
        *avg_bump_angles.tolist(),
    )


# =============================================================================
# OUTPUT / CHECKPOINT
# =============================================================================

def make_result_header(n_timesteps):
    return [
        "normalizing_factor",
        "h_b",
        "v",
        "beta",
        "avg_number_of_bumps",
        "avg_total_active_neurons",
        "avg_bump_width",
        "avg_largest_bump_width",
        "avg_zero_bump_timestep_percentage",
        *[
            f"time{i}_bumpangle"
            for i in range(1, n_timesteps + 1)
        ],
    ]


def combination_key(
    normalizing_factor,
    h_b,
    v,
    beta
):
    """
    Stable key for checkpoint/resume logic.
    """

    return (
        f"{float(normalizing_factor):.2f}|"
        f"{float(h_b):.3f}|"
        f"{float(v):.1f}|"
        f"{int(beta)}"
    )


def load_completed_combinations(
    checkpoint_path
):

    completed = set()

    if not os.path.exists(
        checkpoint_path
    ):

        return completed

    with open(
        checkpoint_path,
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            key = combination_key(
                float(
                    row[
                        "normalizing_factor"
                    ]
                ),
                float(
                    row["h_b"]
                ),
                float(
                    row["v"]
                ),
                int(
                    float(
                        row["beta"]
                    )
                )
            )

            completed.add(
                key
            )

    return completed


def append_checkpoint(
    checkpoint_path,
    result,
    result_header
):
    """
    Write each completed parameter combination immediately.

    If the study is interrupted, completed combinations are preserved.
    """

    file_exists = os.path.exists(
        checkpoint_path
    )

    with open(
        checkpoint_path,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        if not file_exists:

            writer.writerow(
                result_header
            )

        writer.writerow(
            result
        )


def sort_checkpoint_to_final(
    checkpoint_path,
    final_path,
    result_header
):
    """
    Produce a clean, sorted final CSV.
    """

    rows = []

    with open(
        checkpoint_path,
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )

        rows.extend(
            reader
        )

    rows.sort(
        key=lambda row: (
            -float(
                row[
                    "normalizing_factor"
                ]
            ),
            -float(
                row["h_b"]
            ),
            float(
                row["v"]
            ),
            int(
                float(
                    row["beta"]
                )
            )
        )
    )

    with open(
        final_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result_header
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

def save_run_config(
    output_dir,
    input_path,
    db_threshold,
    seeds,
    n_timesteps,
    workers
):

    total_combinations = (
        len(NORMALIZING_FACTORS)
        * len(H_B_VALUES)
        * len(V_VALUES)
        * len(BETA_VALUES)
    )

    config = {

        "input_file":
            os.path.abspath(
                input_path
            ),

        "db_filter":
            f"dB_SPL > {db_threshold}",

        "db_threshold":
            db_threshold,

        "n_retained_doa_timesteps":
            n_timesteps,

        "Ns":
            NS,

        "updates_per_step":
            UPDATES_PER_STEP,

        "normalizing_factors":
            NORMALIZING_FACTORS,

        "h_b_values":
            H_B_VALUES,

        "v_values":
            V_VALUES,

        "beta_values":
            BETA_VALUES,

        "n_seeds":
            len(seeds),

        "master_seed":
            MASTER_SEED,

        "fixed_seeds":
            [
                int(seed)
                for seed in seeds
            ],

        "zero_gap_tolerance":
            ZERO_GAP_TOLERANCE,

        "parallel_workers":
            workers,

        "total_parameter_combinations":
            total_combinations,

        "total_complete_ring_runs":
            (
                total_combinations
                * len(seeds)
            ),

        "implementation":
            (
                "NumPy + multiprocessing, "
                "no Numba"
            ),

        "interaction_optimization":
            (
                "incrementally maintained "
                "J @ spins interaction field"
            ),
    }

    path = os.path.join(
        output_dir,
        "run_config.json"
    )

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    return path


# =============================================================================
# MAIN STUDY
# =============================================================================

def run_study(
    input_path,
    output_dir,
    db_threshold,
    workers
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )
    
    # -------------------------------------------------------------------------
    # Start timing
    # -------------------------------------------------------------------------

    start_time = datetime.now()

    print()
    print("=" * 70)
    print("PARAMETER STUDY STARTED")
    print("=" * 70)
    print(f"Start Time : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------------------------------------------------------
    # Load and normalize input once
    # -------------------------------------------------------------------------

    (
        doa_base,
        timestamps,
        db_values
    ) = load_and_normalize_doa(
        input_path,
        db_threshold
    )

    # -------------------------------------------------------------------------
    # Precompute only 4 interaction matrices
    # -------------------------------------------------------------------------

    print()
    print(
        "Precomputing interaction kernels..."
    )

    kernels = (
        precompute_kernels()
    )

    print(
        "Kernel shape:",
        kernels.shape
    )

    # -------------------------------------------------------------------------
    # Generate fixed 50 seeds
    # -------------------------------------------------------------------------

    seeds = (
        generate_fixed_seeds()
    )

    seed_path = save_seeds(
        seeds,
        output_dir
    )

    # -------------------------------------------------------------------------
    # Save configuration
    # -------------------------------------------------------------------------

    config_path = (
        save_run_config(
            output_dir,
            input_path,
            db_threshold,
            seeds,
            doa_base.shape[0],
            workers
        )
    )

    # -------------------------------------------------------------------------
    # Full factorial parameter grid
    # -------------------------------------------------------------------------

    all_tasks = list(
        itertools.product(
            NORMALIZING_FACTORS,
            H_B_VALUES,
            V_VALUES,
            BETA_VALUES
        )
    )

    total_combinations = len(
        all_tasks
    )

    print()
    print("=" * 70)
    print("PARAMETER STUDY")
    print("=" * 70)

    print(
        f"NORMALIZING_FACTOR values : "
        f"{len(NORMALIZING_FACTORS)}"
    )

    print(
        f"h_b values                : "
        f"{len(H_B_VALUES)}"
    )

    print(
        f"v values                  : "
        f"{len(V_VALUES)}"
    )

    print(
        f"beta values               : "
        f"{len(BETA_VALUES)}"
    )

    print(
        f"Parameter combinations    : "
        f"{total_combinations}"
    )

    print(
        f"Seeds per combination     : "
        f"{len(seeds)}"
    )

    print(
        f"Total complete ring runs  : "
        f"{total_combinations * len(seeds)}"
    )

    print(
        f"DOA timesteps per run     : "
        f"{doa_base.shape[0]}"
    )

    print(
        f"Updates per DOA           : "
        f"{UPDATES_PER_STEP}"
    )

    print(
        f"CPU workers               : "
        f"{workers}"
    )

    print()
    print(
        f"Seeds file                : "
        f"{seed_path}"
    )

    print(
        f"Configuration             : "
        f"{config_path}"
    )

    # -------------------------------------------------------------------------
    # Resume/checkpoint paths
    # -------------------------------------------------------------------------

    checkpoint_path = os.path.join(
        output_dir,
        "parameter_study_checkpoint.csv"
    )

    final_path = os.path.join(
        output_dir,
        "parameter_study_results.csv"
    )

    completed = (
        load_completed_combinations(
            checkpoint_path
        )
    )

    pending_tasks = []

    for task in all_tasks:

        if (
            combination_key(*task)
            not in completed
        ):

            pending_tasks.append(
                task
            )

    print()
    print(
        f"Already completed         : "
        f"{len(completed)}"
    )

    print(
        f"Remaining combinations    : "
        f"{len(pending_tasks)}"
    )

    # -------------------------------------------------------------------------
    # Already complete?
    # -------------------------------------------------------------------------

    if not pending_tasks:

        sort_checkpoint_to_final(
            checkpoint_path,
            final_path,
            result_header
        )

        print()
        print(
            "All parameter combinations "
            "are already complete."
        )

        print(
            f"Final results: "
            f"{final_path}"
        )

        return

    # -------------------------------------------------------------------------
    # Parallel execution
    # -------------------------------------------------------------------------

    print()
    print(
        "Starting parameter sweep..."
    )

    print()

    completed_this_session = 0

    # spawn is safest/most portable, especially on macOS.
    ctx = mp.get_context(
        "spawn"
    )

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            doa_base,
            kernels,
            seeds
        )
    ) as executor:

        future_to_task = {

            executor.submit(
                run_parameter_combination,
                task
            ): task

            for task in pending_tasks
        }

        for future in as_completed(
            future_to_task
        ):

            task = (
                future_to_task[
                    future
                ]
            )

            try:

                result = (
                    future.result()
                )

            except Exception as exc:

                print()
                print(
                    "ERROR while running:"
                )

                print(
                    f"  NORMALIZING_FACTOR = "
                    f"{task[0]}"
                )

                print(
                    f"  h_b = {task[1]}"
                )

                print(
                    f"  v = {task[2]}"
                )

                print(
                    f"  beta = {task[3]}"
                )

                print(
                    f"Exception: {exc}"
                )

                raise
            
            result_header = make_result_header(doa_base.shape[0])

            # Immediately checkpoint.
            append_checkpoint(
                checkpoint_path,
                result,
                result_header
            )

            completed_this_session += 1

            total_done = (
                len(completed)
                + completed_this_session
            )

            percent = (
                100.0
                * total_done
                / total_combinations
            )

            print(
                f"[{total_done:5d}/"
                f"{total_combinations}] "
                f"{percent:6.2f}% | "
                f"NF={result[0]:.2f} | "
                f"h_b={result[1]:.3f} | "
                f"v={result[2]:.1f} | "
                f"beta={int(result[3])} | "
                f"bumps={result[4]:.4f} | "
                f"active={result[5]:.4f} | "
                f"width={result[6]:.4f} | "
                f"largest={result[7]:.4f} | "
                f"zero={result[8]:.2f}%",
                flush=True
            )

    # -------------------------------------------------------------------------
    # Final sorted output
    # -------------------------------------------------------------------------

    sort_checkpoint_to_final(
        checkpoint_path,
        final_path,
        result_header
    )

    print()
    print("=" * 70)
    print("PARAMETER STUDY COMPLETE")
    print("=" * 70)

    print(
        f"Final results : "
        f"{final_path}"
    )

    print(
        f"Checkpoint    : "
        f"{checkpoint_path}"
    )

    print(
        f"Seeds         : "
        f"{seed_path}"
    )

    print(
        f"Configuration : "
        f"{config_path}"
    )

        
    # -------------------------------------------------------------------------
    # End timing
    # -------------------------------------------------------------------------

    end_time = datetime.now()
    elapsed = end_time - start_time

    print()
    print("=" * 70)
    print("TIMING")
    print("=" * 70)
    print(f"Start Time : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    days = elapsed.days
    hours, remainder = divmod(elapsed.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f"Elapsed    : "
        f"{days}d "
        f"{hours:02d}h "
        f"{minutes:02d}m "
        f"{seconds:02d}s"
    )



# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Ring attractor full factorial "
            "parameter study without Numba."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to one DOA collection CSV."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "./ring_parameter_study"
        ),
        help=(
            "Output directory."
        )
    )

    parser.add_argument(
        "--db-threshold",
        type=float,
        default=50.0,
        help=(
            "Only rows with "
            "dB_SPL > threshold "
            "are processed."
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(
            1,
            (os.cpu_count() or 2) - 1
        ),
        help=(
            "Number of multiprocessing "
            "worker processes."
        )
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    mp.freeze_support()

    args = parse_args()

    run_study(
        input_path=args.input,
        output_dir=args.output_dir,
        db_threshold=args.db_threshold,
        workers=args.workers
    )