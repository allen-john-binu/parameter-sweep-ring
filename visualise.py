import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import (
    Dash,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    html,
    no_update,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Script directory.
#
# Expected structure:
#
# parent_directory/
# |
# |-- this_script.py
# |
# |-- ztParameterStudy1/
# |   |-- parameter_study_results.csv
# |   `-- run_config.json
# |
# |-- ztParameterStudy2/
# |   |-- parameter_study_results.csv
# |   `-- run_config.json
# |
# `-- ztParameterStudy3/
#     |-- parameter_study_results.csv
#     `-- run_config.json

BASE_DIR = Path(__file__).resolve().parent

STUDY_PREFIX = "ztParameterStudy"

RESULTS_FILENAME = "parameter_study_results.csv"

CONFIG_FILENAME = "run_config.json"


# =============================================================================
# METRICS
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
        "avg_zero_bump_timestep_percentage",
        "Average Zero-Bump Timesteps (%)",
    ),
]


METRIC_COLUMNS = [
    metric
    for metric, _ in METRICS
]


# =============================================================================
# REQUIRED CSV COLUMNS
# =============================================================================

REQUIRED_COLUMNS = [
    "normalizing_factor",
    "h_b",
    "v",
    "beta",
    "avg_number_of_bumps",
    "avg_total_active_neurons",
    "avg_bump_width",
    "avg_zero_bump_timestep_percentage",
]


# =============================================================================
# STUDY DISCOVERY
# =============================================================================

def discover_studies():
    """
    Automatically discover parameter-study folders.

    A valid study folder must:

        1. Be a directory.
        2. Start with STUDY_PREFIX.
        3. Contain parameter_study_results.csv.

    Examples:

        ztParameterStudy1/
        ztParameterStudy2/
        ztParameterStudy3/
    """

    studies = []

    for path in BASE_DIR.iterdir():

        if not path.is_dir():
            continue

        if not path.name.startswith(
            STUDY_PREFIX
        ):
            continue

        results_path = (
            path
            / RESULTS_FILENAME
        )

        if not results_path.exists():
            continue

        studies.append(path)

    # -------------------------------------------------------------------------
    # Natural-ish sorting
    #
    # ztParameterStudy1
    # ztParameterStudy2
    # ztParameterStudy10
    # -------------------------------------------------------------------------

    def sort_key(path):

        suffix = path.name.replace(
            STUDY_PREFIX,
            "",
        )

        try:

            return (
                0,
                int(suffix),
            )

        except ValueError:

            return (
                1,
                path.name,
            )

    studies.sort(
        key=sort_key
    )

    return studies


# =============================================================================
# LOAD STUDY DATA
# =============================================================================

def load_study_dataframe(
    study_name,
):
    """
    Load and validate parameter_study_results.csv
    for one study.

    The CSV is the source of truth for all parameter values.
    """

    study_path = (
        BASE_DIR
        / study_name
    )

    csv_path = (
        study_path
        / RESULTS_FILENAME
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Results file not found:\n"
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path
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
            f"Study '{study_name}' is missing "
            f"required columns:\n"
            f"{missing}"
        )

    # -------------------------------------------------------------------------
    # Convert required columns to numeric
    #
    # This avoids problems caused by CSV columns accidentally being loaded
    # as strings.
    # -------------------------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Remove rows where parameter coordinates are invalid
    # -------------------------------------------------------------------------

    df = df.dropna(
        subset=[
            "normalizing_factor",
            "h_b",
            "v",
            "beta",
        ]
    ).copy()

    return df


# =============================================================================
# LOAD OPTIONAL JSON CONFIG
# =============================================================================

def load_study_config(
    study_name,
):
    """
    Load run_config.json if available.

    IMPORTANT:

    JSON is NOT used to determine available parameter values.

    CSV is the source of truth for:

        - v
        - beta
        - normalizing_factor
        - h_b

    JSON is used only for optional metadata display.
    """

    config_path = (
        BASE_DIR
        / study_name
        / CONFIG_FILENAME
    )

    if not config_path.exists():
        return None

    try:

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    except Exception as exc:

        print(
            f"[WARNING] Could not read "
            f"{config_path}: {exc}"
        )

        return None


# =============================================================================
# NUMBER FORMATTING
# =============================================================================

def format_number(
    value,
    decimals=4,
):
    """
    Format numeric values cleanly.

    Examples:

        50.0    -> 50
        0.3     -> 0.3
        12.3456 -> 12.3456
    """

    if value is None:
        return "N/A"

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(
            value
        )

    if np.isnan(value):
        return "NaN"

    if value.is_integer():

        return str(
            int(value)
        )

    return (
        f"{value:.{decimals}f}"
        .rstrip("0")
        .rstrip(".")
    )


# =============================================================================
# AGGREGATE ONE (v, beta) PAIR
# =============================================================================

def aggregate_pair_dataframe(
    pair_df,
):
    """
    Collapse duplicate (normalizing_factor, h_b) combinations.

    Why this is needed
    ------------------

    Some CSV files may contain more than one row for the same:

        v
        beta
        normalizing_factor
        h_b

    Instead of failing with:

        "Duplicate combinations found"

    we combine those rows.

    The four metric columns are averaged.

    A new column:

        source_row_count

    records how many original CSV rows contributed to each
    heatmap cell.

    If there are no duplicates, source_row_count = 1.
    """

    if pair_df.empty:

        return pair_df.copy()

    # -------------------------------------------------------------------------
    # Group by the heatmap coordinates
    # -------------------------------------------------------------------------

    grouped = (
        pair_df
        .groupby(
            [
                "normalizing_factor",
                "h_b",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                "v": "first",
                "beta": "first",

                "avg_number_of_bumps":
                    "mean",

                "avg_total_active_neurons":
                    "mean",

                "avg_bump_width":
                    "mean",

                "avg_zero_bump_timestep_percentage":
                    "mean",
            }
        )
    )

    # -------------------------------------------------------------------------
    # Count how many original rows contributed to every cell
    # -------------------------------------------------------------------------

    counts = (
        pair_df
        .groupby(
            [
                "normalizing_factor",
                "h_b",
            ],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "source_row_count"
            }
        )
    )

    grouped = grouped.merge(
        counts,
        on=[
            "normalizing_factor",
            "h_b",
        ],
        how="left",
    )

    # -------------------------------------------------------------------------
    # Sort consistently
    # -------------------------------------------------------------------------

    grouped = (
        grouped
        .sort_values(
            [
                "h_b",
                "normalizing_factor",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return grouped


# =============================================================================
# EMPTY FIGURE
# =============================================================================

def create_empty_figure(
    message,
):
    """
    Create a blank Plotly figure with a message.
    """

    figure = go.Figure()

    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=18,
        ),
    )

    figure.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(
            l=50,
            r=30,
            t=60,
            b=50,
        ),
        xaxis=dict(
            visible=False,
        ),
        yaxis=dict(
            visible=False,
        ),
    )

    return figure


# =============================================================================
# CREATE ONE HEATMAP
# =============================================================================

def create_heatmap(
    pair_df,
    metric,
    title,
    selected_point=None,
):
    """
    Create one interactive heatmap.

    X-axis:
        normalizing_factor

    Y-axis:
        h_b

    COLOR SCALE:
        LOCAL.

    Each heatmap automatically determines its color scale
    from the selected (v, beta) pair only.

    No global zmin/zmax is used.

    Therefore:

        changing v or beta
        -> changes pair_df
        -> color scale recalculates locally

    Each of the four metrics also gets its own independent
    local scale.
    """

    # -------------------------------------------------------------------------
    # Pivot metric data
    #
    # pair_df has already been aggregated, so every
    # (normalizing_factor, h_b) pair is unique.
    # -------------------------------------------------------------------------

    heatmap = pair_df.pivot(
        index="h_b",
        columns="normalizing_factor",
        values=metric,
    )

    heatmap = heatmap.sort_index(
        ascending=True
    )

    heatmap = heatmap.sort_index(
        axis=1,
        ascending=True,
    )

    x_values = (
        heatmap
        .columns
        .to_numpy(
            dtype=float
        )
    )

    y_values = (
        heatmap
        .index
        .to_numpy(
            dtype=float
        )
    )

    z_values = heatmap.to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Create custom data
    #
    # customdata[row, column]:
    #
    #     [
    #         normalizing_factor,
    #         h_b
    #     ]
    #
    # This ensures clicking a heatmap cell gives us the exact
    # parameter coordinates.
    # -------------------------------------------------------------------------

    customdata = np.empty(
        (
            len(y_values),
            len(x_values),
            2,
        ),
        dtype=float,
    )

    for row_index, h_b in enumerate(
        y_values
    ):

        for column_index, nf in enumerate(
            x_values
        ):

            customdata[
                row_index,
                column_index,
                0,
            ] = nf

            customdata[
                row_index,
                column_index,
                1,
            ] = h_b

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    figure = go.Figure()

    # -------------------------------------------------------------------------
    # Heatmap
    #
    # IMPORTANT:
    #
    # zmin and zmax are intentionally NOT specified.
    #
    # Plotly therefore automatically scales the colors using
    # this selected pair's metric values.
    #
    # This gives LOCAL color scaling.
    # -------------------------------------------------------------------------

    figure.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            customdata=customdata,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(
                    text=title,
                ),
            ),
            hovertemplate=(
                "<b>"
                + title
                + "</b>"
                "<br><br>"
                "NORMALIZING_FACTOR: "
                "%{customdata[0]:.4g}"
                "<br>"
                "h_b: "
                "%{customdata[1]:.4g}"
                "<br>"
                + title
                + ": %{z:.6g}"
                "<extra></extra>"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Highlight selected point
    # -------------------------------------------------------------------------

    if selected_point is not None:

        selected_nf = float(
            selected_point[
                "normalizing_factor"
            ]
        )

        selected_hb = float(
            selected_point[
                "h_b"
            ]
        )

        # ---------------------------------------------------------------------
        # Large open square
        # ---------------------------------------------------------------------

        figure.add_trace(
            go.Scatter(
                x=[
                    selected_nf
                ],
                y=[
                    selected_hb
                ],
                mode="markers",
                marker=dict(
                    symbol="square-open",
                    size=30,
                    color="white",
                    line=dict(
                        color="black",
                        width=4,
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # ---------------------------------------------------------------------
        # Center X
        # ---------------------------------------------------------------------

        figure.add_trace(
            go.Scatter(
                x=[
                    selected_nf
                ],
                y=[
                    selected_hb
                ],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=13,
                    color="white",
                    line=dict(
                        color="black",
                        width=2,
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    figure.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=20,
            ),
        ),
        template="plotly_white",
        height=540,
        margin=dict(
            l=80,
            r=50,
            t=80,
            b=80,
        ),
        clickmode="event+select",
    )

    # -------------------------------------------------------------------------
    # X axis
    # -------------------------------------------------------------------------

    figure.update_xaxes(
        title="NORMALIZING_FACTOR",
        tickmode="array",
        tickvals=x_values,
        ticktext=[
            format_number(x)
            for x in x_values
        ],
        tickangle=-45,
    )

    # -------------------------------------------------------------------------
    # Y axis
    # -------------------------------------------------------------------------

    figure.update_yaxes(
        title="h_b",
        tickmode="array",
        tickvals=y_values,
        ticktext=[
            format_number(y)
            for y in y_values
        ],
    )

    return figure


# =============================================================================
# SELECT EXACT PARAMETER ROW
# =============================================================================

def find_selected_row(
    pair_df,
    normalizing_factor,
    h_b,
):
    """
    Find the aggregated row corresponding to a selected heatmap cell.

    np.isclose() is used for floating-point parameter matching.
    """

    selected = pair_df[
        np.isclose(
            pair_df[
                "normalizing_factor"
            ],
            normalizing_factor,
        )
        &
        np.isclose(
            pair_df[
                "h_b"
            ],
            h_b,
        )
    ]

    if selected.empty:
        return None

    return selected.iloc[0]


# =============================================================================
# VALUE CARD
# =============================================================================

def create_value_card(
    label,
    value,
):
    """
    Small reusable dashboard card.
    """

    return html.Div(
        [
            html.Div(
                label,
                style={
                    "fontSize":
                        "13px",

                    "fontWeight":
                        "bold",

                    "marginBottom":
                        "7px",

                    "color":
                        "#555",
                },
            ),

            html.Div(
                value,
                style={
                    "fontSize":
                        "22px",

                    "fontWeight":
                        "bold",
                },
            ),
        ],
        style={
            "padding":
                "15px",

            "border":
                "1px solid #ddd",

            "borderRadius":
                "8px",

            "backgroundColor":
                "#fafafa",
        },
    )


# =============================================================================
# CREATE SELECTED-POINT DETAILS PANEL
# =============================================================================

def create_selected_point_panel(
    selected_row,
):
    """
    Create the information panel below the heatmaps.
    """

    if selected_row is None:

        return html.Div(
            [
                html.H3(
                    "Selected Parameter Point"
                ),

                html.P(
                    (
                        "Click any cell in any heatmap. "
                        "The same parameter point will be "
                        "highlighted across all four plots."
                    )
                ),
            ]
        )

    source_count = int(
        selected_row.get(
            "source_row_count",
            1,
        )
    )

    children = [
        html.H3(
            "Selected Parameter Point",
            style={
                "marginTop":
                    "0",
            },
        ),

        # ---------------------------------------------------------------------
        # Parameter coordinates
        # ---------------------------------------------------------------------

        html.Div(
            [
                create_value_card(
                    "NORMALIZING_FACTOR",
                    format_number(
                        selected_row[
                            "normalizing_factor"
                        ]
                    ),
                ),

                create_value_card(
                    "h_b",
                    format_number(
                        selected_row[
                            "h_b"
                        ]
                    ),
                ),

                create_value_card(
                    "v",
                    format_number(
                        selected_row[
                            "v"
                        ]
                    ),
                ),

                create_value_card(
                    "beta",
                    format_number(
                        selected_row[
                            "beta"
                        ]
                    ),
                ),
            ],
            style={
                "display":
                    "grid",

                "gridTemplateColumns":
                    (
                        "repeat("
                        "auto-fit, "
                        "minmax(180px, 1fr)"
                        ")"
                    ),

                "gap":
                    "12px",

                "marginBottom":
                    "20px",
            },
        ),

        # ---------------------------------------------------------------------
        # Metric values
        # ---------------------------------------------------------------------

        html.Div(
            [
                create_value_card(
                    "Average Number of Bumps",
                    format_number(
                        selected_row[
                            "avg_number_of_bumps"
                        ],
                        decimals=6,
                    ),
                ),

                create_value_card(
                    "Average Total Active Neurons",
                    format_number(
                        selected_row[
                            "avg_total_active_neurons"
                        ],
                        decimals=6,
                    ),
                ),

                create_value_card(
                    "Average Bump Width",
                    format_number(
                        selected_row[
                            "avg_bump_width"
                        ],
                        decimals=6,
                    ),
                ),

                create_value_card(
                    "Average Zero-Bump Timesteps (%)",
                    format_number(
                        selected_row[
                            "avg_zero_bump_timestep_percentage"
                        ],
                        decimals=7,
                    ) + "%",
                ),
            ],
            style={
                "display":
                    "grid",

                "gridTemplateColumns":
                    (
                        "repeat("
                        "auto-fit, "
                        "minmax(220px, 1fr)"
                        ")"
                    ),

                "gap":
                    "12px",
            },
        ),
    ]

    # -------------------------------------------------------------------------
    # Show aggregation notice only when multiple source rows contributed
    # -------------------------------------------------------------------------

    if source_count > 1:

        children.append(
            html.Div(
                (
                    f"Note: {source_count} CSV rows existed for this "
                    "(normalizing_factor, h_b, v, beta) combination. "
                    "The displayed metric values are their means."
                ),
                style={
                    "marginTop":
                        "18px",

                    "padding":
                        "12px",

                    "backgroundColor":
                        "#fff8e1",

                    "border":
                        "1px solid #ffe082",

                    "borderRadius":
                        "6px",
                },
            )
        )

    return html.Div(
        children
    )


# =============================================================================
# STUDY INFORMATION PANEL
# =============================================================================

def create_study_info(
    study_name,
    df,
):
    """
    Display information about the selected study.
    """

    config = load_study_config(
        study_name
    )

    unique_v = np.sort(
        df[
            "v"
        ]
        .dropna()
        .unique()
    )

    unique_beta = np.sort(
        df[
            "beta"
        ]
        .dropna()
        .unique()
    )

    normalizing_factors = np.sort(
        df[
            "normalizing_factor"
        ]
        .dropna()
        .unique()
    )

    h_b_values = np.sort(
        df[
            "h_b"
        ]
        .dropna()
        .unique()
    )

    # -------------------------------------------------------------------------
    # Count actual existing (v, beta) pairs
    # -------------------------------------------------------------------------

    pair_count = len(
        df[
            [
                "v",
                "beta",
            ]
        ]
        .drop_duplicates()
    )

    children = [
        html.Strong(
            study_name
        ),

        html.Span(
            (
                f" | Rows: {len(df)}"
                f" | v values: {len(unique_v)}"
                f" | beta values: {len(unique_beta)}"
                f" | existing (v, beta) pairs: {pair_count}"
                f" | normalizing factors: "
                f"{len(normalizing_factors)}"
                f" | h_b values: "
                f"{len(h_b_values)}"
            )
        ),
    ]

    # -------------------------------------------------------------------------
    # Optional JSON metadata
    # -------------------------------------------------------------------------

    if config is not None:

        metadata = []

        if "n_seeds" in config:

            metadata.append(
                f"Seeds: "
                f"{config['n_seeds']}"
            )

        if "Ns" in config:

            metadata.append(
                f"Ns: "
                f"{config['Ns']}"
            )

        if "updates_per_step" in config:

            metadata.append(
                (
                    "Updates/step: "
                    f"{config['updates_per_step']}"
                )
            )

        if metadata:

            children.extend(
                [
                    html.Br(),

                    html.Span(
                        " | ".join(
                            metadata
                        )
                    ),
                ]
            )

    return html.Div(
        children,
        style={
            "padding":
                "12px 16px",

            "backgroundColor":
                "#f5f5f5",

            "borderRadius":
                "8px",

            "marginTop":
                "15px",
        },
    )


# =============================================================================
# DISCOVER INITIAL STUDIES
# =============================================================================

STUDY_PATHS = (
    discover_studies()
)

STUDY_NAMES = [
    path.name
    for path in STUDY_PATHS
]


# =============================================================================
# DASH APPLICATION
# =============================================================================

app = Dash(
    __name__
)

app.title = (
    "Ring Attractor Parameter Study Explorer"
)


# =============================================================================
# PAGE LAYOUT
# =============================================================================

app.layout = html.Div(
    [
        # ---------------------------------------------------------------------
        # Header
        # ---------------------------------------------------------------------

        html.H1(
            "Ring Attractor Parameter Study Explorer",
            style={
                "textAlign":
                    "center",

                "marginBottom":
                    "5px",
            },
        ),

        html.P(
            (
                "Select a parameter study, choose a valid "
                "(v, beta) pair, and explore the four "
                "metrics interactively."
            ),
            style={
                "textAlign":
                    "center",

                "color":
                    "#666",

                "marginTop":
                    "0",

                "marginBottom":
                    "30px",
            },
        ),

        # ---------------------------------------------------------------------
        # Controls
        # ---------------------------------------------------------------------

        html.Div(
            [
                # -------------------------------------------------------------
                # Study dropdown
                # -------------------------------------------------------------

                html.Div(
                    [
                        html.Label(
                            "Parameter Study",
                            style={
                                "fontWeight":
                                    "bold",
                            },
                        ),

                        dcc.Dropdown(
                            id="study-dropdown",

                            options=[
                                {
                                    "label":
                                        study,

                                    "value":
                                        study,
                                }
                                for study
                                in STUDY_NAMES
                            ],

                            value=(
                                STUDY_NAMES[0]
                                if STUDY_NAMES
                                else None
                            ),

                            clearable=False,

                            placeholder=(
                                "Select a study"
                            ),
                        ),
                    ]
                ),

                # -------------------------------------------------------------
                # v dropdown
                # -------------------------------------------------------------

                html.Div(
                    [
                        html.Label(
                            "v",
                            style={
                                "fontWeight":
                                    "bold",
                            },
                        ),

                        dcc.Dropdown(
                            id="v-dropdown",

                            clearable=False,

                            placeholder=(
                                "Select v"
                            ),
                        ),
                    ]
                ),

                # -------------------------------------------------------------
                # beta dropdown
                # -------------------------------------------------------------

                html.Div(
                    [
                        html.Label(
                            "beta",
                            style={
                                "fontWeight":
                                    "bold",
                            },
                        ),

                        dcc.Dropdown(
                            id="beta-dropdown",

                            clearable=False,

                            placeholder=(
                                "Select beta"
                            ),
                        ),
                    ]
                ),
            ],
            style={
                "display":
                    "grid",

                "gridTemplateColumns":
                    "2fr 1fr 1fr",

                "gap":
                    "20px",

                "alignItems":
                    "end",
            },
        ),

        # ---------------------------------------------------------------------
        # Study metadata
        # ---------------------------------------------------------------------

        html.Div(
            id="study-info"
        ),

        # ---------------------------------------------------------------------
        # Selected point storage
        # ---------------------------------------------------------------------

        dcc.Store(
            id="selected-point-store",
            data=None,
        ),

        # ---------------------------------------------------------------------
        # Graphs
        # ---------------------------------------------------------------------

        html.Div(
            [
                dcc.Graph(
                    id="graph-bumps",
                    config={
                        "displaylogo":
                            False,

                        "responsive":
                            True,
                    },
                ),

                dcc.Graph(
                    id="graph-active-neurons",
                    config={
                        "displaylogo":
                            False,

                        "responsive":
                            True,
                    },
                ),

                dcc.Graph(
                    id="graph-bump-width",
                    config={
                        "displaylogo":
                            False,

                        "responsive":
                            True,
                    },
                ),

                dcc.Graph(
                    id="graph-zero-bump-percentage",
                    config={
                        "displaylogo":
                            False,

                        "responsive":
                            True,
                    },
                ),
            ],
            style={
                "display":
                    "grid",

                "gridTemplateColumns":
                    "repeat(2, minmax(0, 1fr))",

                "gap":
                    "15px",

                "marginTop":
                    "25px",
            },
        ),

        # ---------------------------------------------------------------------
        # Selected point information
        # ---------------------------------------------------------------------

        html.Div(
            id="selected-point-panel",

            children=(
                create_selected_point_panel(
                    None
                )
            ),

            style={
                "marginTop":
                    "25px",

                "padding":
                    "20px",

                "border":
                    "1px solid #ddd",

                "borderRadius":
                    "10px",

                "marginBottom":
                    "40px",
            },
        ),
    ],
    style={
        "maxWidth":
            "1700px",

        "margin":
            "0 auto",

        "padding":
            "25px",

        "fontFamily":
            "Arial, Helvetica, sans-serif",
    },
)


# =============================================================================
# CALLBACK:
# STUDY -> AVAILABLE v VALUES
# =============================================================================

@app.callback(
    Output(
        "v-dropdown",
        "options",
    ),

    Output(
        "v-dropdown",
        "value",
    ),

    Output(
        "study-info",
        "children",
    ),

    Input(
        "study-dropdown",
        "value",
    ),
)
def update_study(
    study_name,
):
    """
    Selecting a study:

        1. Loads the CSV.
        2. Automatically discovers v values from the CSV.
        3. Populates the v dropdown.
        4. Selects the first available v.
        5. Updates study metadata.

    beta is handled by another callback because valid beta
    values depend on the selected v.
    """

    if study_name is None:

        return (
            [],
            None,
            html.Div(
                "No study selected."
            ),
        )

    df = load_study_dataframe(
        study_name
    )

    # -------------------------------------------------------------------------
    # Automatically discover v
    # -------------------------------------------------------------------------

    v_values = np.sort(
        df[
            "v"
        ]
        .dropna()
        .unique()
    )

    v_options = [
        {
            "label":
                format_number(
                    value
                ),

            "value":
                float(value),
        }
        for value
        in v_values
    ]

    default_v = (
        float(
            v_values[0]
        )
        if len(v_values) > 0
        else None
    )

    study_info = (
        create_study_info(
            study_name,
            df,
        )
    )

    return (
        v_options,
        default_v,
        study_info,
    )


# =============================================================================
# CALLBACK:
# STUDY + SELECTED v -> VALID beta VALUES
# =============================================================================

@app.callback(
    Output(
        "beta-dropdown",
        "options",
    ),

    Output(
        "beta-dropdown",
        "value",
    ),

    Input(
        "study-dropdown",
        "value",
    ),

    Input(
        "v-dropdown",
        "value",
    ),
)
def update_beta_values(
    study_name,
    v_value,
):
    """
    Automatically discover beta values that actually exist
    for the selected v.

    This makes parameter discovery pair-aware.

    Example:

        CSV contains:

            v=0.3, beta=50
            v=0.3, beta=100

            v=0.4, beta=200
            v=0.4, beta=400

        If v=0.3:

            beta dropdown -> 50, 100

        If v=0.4:

            beta dropdown -> 200, 400

    Invalid (v, beta) combinations therefore cannot normally
    be selected.
    """

    if (
        study_name is None
        or v_value is None
    ):

        return (
            [],
            None,
        )

    df = load_study_dataframe(
        study_name
    )

    # -------------------------------------------------------------------------
    # Select rows matching v
    # -------------------------------------------------------------------------

    v_df = df[
        np.isclose(
            df[
                "v"
            ],
            float(
                v_value
            ),
        )
    ]

    # -------------------------------------------------------------------------
    # Discover beta values only from those rows
    # -------------------------------------------------------------------------

    beta_values = np.sort(
        v_df[
            "beta"
        ]
        .dropna()
        .unique()
    )

    beta_options = [
        {
            "label":
                format_number(
                    value
                ),

            "value":
                float(value),
        }
        for value
        in beta_values
    ]

    default_beta = (
        float(
            beta_values[0]
        )
        if len(beta_values) > 0
        else None
    )

    return (
        beta_options,
        default_beta,
    )


# =============================================================================
# CALLBACK:
# CLEAR SELECTED POINT WHEN STUDY / v / beta CHANGES
# =============================================================================

@app.callback(
    Output(
        "selected-point-store",
        "data",
        allow_duplicate=True,
    ),

    Input(
        "study-dropdown",
        "value",
    ),

    Input(
        "v-dropdown",
        "value",
    ),

    Input(
        "beta-dropdown",
        "value",
    ),

    prevent_initial_call=True,
)
def clear_selected_point(
    study_name,
    v_value,
    beta_value,
):
    """
    A selected point belongs to one specific:

        study
        v
        beta

    Clear it whenever any of those changes.
    """

    return None



# =============================================================================
# CALLBACK:
# CLICK ANY HEATMAP -> STORE SELECTED PARAMETER POINT
# =============================================================================

@app.callback(
    Output(
        "selected-point-store",
        "data",
    ),
    Input(
        "graph-bumps",
        "clickData",
    ),
    Input(
        "graph-active-neurons",
        "clickData",
    ),
    Input(
        "graph-bump-width",
        "clickData",
    ),
    Input(
        "graph-zero-bump-percentage",
        "clickData",
    ),
    prevent_initial_call=True,
)
def select_parameter_point(
    click_bumps,
    click_active,
    click_width,
    click_zero,
):
    """
    Clicking any heatmap stores the selected:

        normalizing_factor
        h_b

    The same point is then highlighted across all four plots
    and shown in the selected-point details panel.
    """

    # -------------------------------------------------------------------------
    # Determine which graph triggered the callback
    # -------------------------------------------------------------------------

    triggered = callback_context.triggered

    if not triggered:
        return no_update

    triggered_id = (
        triggered[0]["prop_id"]
        .split(".")[0]
    )

    click_map = {
        "graph-bumps":
            click_bumps,

        "graph-active-neurons":
            click_active,

        "graph-bump-width":
            click_width,

        "graph-zero-percentage":
            click_zero,
    }

    click_data = click_map.get(
        triggered_id
    )

    # -------------------------------------------------------------------------
    # Validate click data
    # -------------------------------------------------------------------------

    if (
        click_data is None
        or "points" not in click_data
        or len(click_data["points"]) == 0
    ):
        return no_update

    point = click_data["points"][0]

    # -------------------------------------------------------------------------
    # First try customdata
    # -------------------------------------------------------------------------

    customdata = point.get(
        "customdata"
    )

    if (
        customdata is not None
        and len(customdata) >= 2
    ):

        try:

            normalizing_factor = float(
                customdata[0]
            )

            h_b = float(
                customdata[1]
            )

            return {
                "normalizing_factor":
                    normalizing_factor,

                "h_b":
                    h_b,
            }

        except (
            TypeError,
            ValueError,
            IndexError,
        ):

            pass

    # -------------------------------------------------------------------------
    # Fallback:
    #
    # Plotly heatmap clickData normally also contains x and y.
    #
    # x = normalizing_factor
    # y = h_b
    #
    # This makes selection work even if customdata is missing for some reason.
    # -------------------------------------------------------------------------

    x_value = point.get(
        "x"
    )

    y_value = point.get(
        "y"
    )

    if (
        x_value is None
        or y_value is None
    ):
        return no_update

    try:

        return {
            "normalizing_factor":
                float(x_value),

            "h_b":
                float(y_value),
        }

    except (
        TypeError,
        ValueError,
    ):

        return no_update
# =============================================================================
# CALLBACK:
# UPDATE ALL FOUR HEATMAPS
# =============================================================================

@app.callback(
    Output(
        "graph-bumps",
        "figure",
    ),

    Output(
        "graph-active-neurons",
        "figure",
    ),

    Output(
        "graph-bump-width",
        "figure",
    ),

    Output(
        "graph-zero-bump-percentage",
        "figure",
    ),

    Output(
        "selected-point-panel",
        "children",
    ),

    Input(
        "study-dropdown",
        "value",
    ),

    Input(
        "v-dropdown",
        "value",
    ),

    Input(
        "beta-dropdown",
        "value",
    ),

    Input(
        "selected-point-store",
        "data",
    ),
)
def update_heatmaps(
    study_name,
    v_value,
    beta_value,
    selected_point,
):
    """
    Render four linked heatmaps.

    Workflow:

        Study CSV
            |
            v
        select v
            |
            v
        select valid beta
            |
            v
        filter exact (v, beta)
            |
            v
        aggregate duplicate
        (normalizing_factor, h_b)
            |
            v
        create four heatmaps
            |
            v
        LOCAL color scaling
        independently for each metric
    """

    # -------------------------------------------------------------------------
    # Missing selection
    # -------------------------------------------------------------------------

    if (
        study_name is None
        or v_value is None
        or beta_value is None
    ):

        empty = (
            create_empty_figure(
                "Select a study, v, and beta."
            )
        )

        return (
            empty,
            empty,
            empty,
            empty,
            create_selected_point_panel(
                None
            ),
        )

    # -------------------------------------------------------------------------
    # Load study
    # -------------------------------------------------------------------------

    df = load_study_dataframe(
        study_name
    )

    # -------------------------------------------------------------------------
    # Filter exact selected (v, beta) pair
    #
    # np.isclose handles floating-point representation safely.
    # -------------------------------------------------------------------------

    pair_df_raw = df[
        np.isclose(
            df[
                "v"
            ],
            float(
                v_value
            ),
        )
        &
        np.isclose(
            df[
                "beta"
            ],
            float(
                beta_value
            ),
        )
    ].copy()

    # -------------------------------------------------------------------------
    # No matching pair
    # -------------------------------------------------------------------------

    if pair_df_raw.empty:

        empty = (
            create_empty_figure(
                (
                    "No data found for "
                    f"v={format_number(v_value)}, "
                    f"beta={format_number(beta_value)}"
                )
            )
        )

        return (
            empty,
            empty,
            empty,
            empty,
            create_selected_point_panel(
                None
            ),
        )

    # -------------------------------------------------------------------------
    # Detect duplicates for logging
    #
    # We do NOT reject them anymore.
    # -------------------------------------------------------------------------

    duplicate_mask = (
        pair_df_raw
        .duplicated(
            subset=[
                "normalizing_factor",
                "h_b",
            ],
            keep=False,
        )
    )

    duplicate_row_count = int(
        duplicate_mask.sum()
    )

    if duplicate_row_count > 0:

        duplicate_cell_count = (
            pair_df_raw.loc[
                duplicate_mask,
                [
                    "normalizing_factor",
                    "h_b",
                ],
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(
            (
                f"[INFO] "
                f"{study_name}: "
                f"v={format_number(v_value)}, "
                f"beta={format_number(beta_value)} "
                f"contains {duplicate_row_count} rows "
                f"across {duplicate_cell_count} duplicated "
                "(normalizing_factor, h_b) cells. "
                "Metric values will be averaged."
            )
        )

    # -------------------------------------------------------------------------
    # Aggregate duplicate heatmap coordinates
    # -------------------------------------------------------------------------

    pair_df = (
        aggregate_pair_dataframe(
            pair_df_raw
        )
    )

    # -------------------------------------------------------------------------
    # Create all four figures
    #
    # IMPORTANT:
    #
    # create_heatmap receives ONLY pair_df for the currently selected
    # (v, beta).
    #
    # No global zmin/zmax is passed.
    #
    # Therefore every metric gets LOCAL Plotly color scaling.
    # -------------------------------------------------------------------------

    figures = []

    for metric, title in METRICS:

        figure = create_heatmap(
            pair_df=pair_df,
            metric=metric,
            title=title,
            selected_point=selected_point,
        )

        figures.append(
            figure
        )

    # -------------------------------------------------------------------------
    # Selected-point details
    # -------------------------------------------------------------------------

    selected_row = None

    if selected_point is not None:

        selected_row = (
            find_selected_row(
                pair_df=pair_df,

                normalizing_factor=float(
                    selected_point[
                        "normalizing_factor"
                    ]
                ),

                h_b=float(
                    selected_point[
                        "h_b"
                    ]
                ),
            )
        )

    panel = (
        create_selected_point_panel(
            selected_row
        )
    )

    return (
        figures[0],
        figures[1],
        figures[2],
        figures[3],
        panel,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 80
    )

    print(
        "RING ATTRACTOR PARAMETER STUDY EXPLORER"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Base directory:"
        f"\n    {BASE_DIR}"
    )

    print()

    print(
        f"Discovered "
        f"{len(STUDY_NAMES)} "
        f"parameter studies:"
    )

    for study_name in STUDY_NAMES:

        print(
            f"    - {study_name}"
        )

    if not STUDY_NAMES:

        print()

        print(
            "[WARNING] No parameter-study "
            "folders were discovered."
        )

        print(
            f"Expected folders starting with: "
            f"{STUDY_PREFIX}"
        )

        print(
            f"Each folder must contain: "
            f"{RESULTS_FILENAME}"
        )

    print()

    print(
        "Starting local dashboard..."
    )

    print(
        "Open the address shown below "
        "in your browser."
    )

    print()

    app.run(
        debug=True,
    )