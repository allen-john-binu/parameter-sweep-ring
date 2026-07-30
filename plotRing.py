import numpy as np
import matplotlib.pyplot as plt


def plot_ring(spin_string, save_path=None):
    """
    Parameters
    ----------
    spin_string : str
        Binary string of length 120, e.g.
        "0000011111000..."
    save_path : str or None
        If given, save figure to this path.
    """

    N = 120

    if len(spin_string) != N:
        raise ValueError(f"Expected {N} spins, got {len(spin_string)}")

    spins = np.array([int(c) for c in spin_string])

    # Preferred neuron angles:
    # -180 deg at left, increasing clockwise.
    #
    # We first define mathematical angles in radians
    # running from pi to -pi.
    thetas = np.linspace(np.pi, -np.pi, N, endpoint=False)

    # Circle coordinates
    R = 1.0
    x = R * np.cos(thetas)
    y = R * np.sin(thetas)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw neurons
    colors = np.where(spins == 1, "red", "black")

    ax.scatter(
        x,
        y,
        s=70,
        c=colors,
        edgecolors="none",
        zorder=3,
    )

    # Draw bump direction if there are active neurons
    if np.any(spins == 1):

        phi = np.angle(np.sum(np.exp(1j * thetas[spins == 1])))

        ax.annotate(
            "",
            xy=(0.78 * np.cos(phi), 0.78 * np.sin(phi)),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=2.5,
                color="black",
                mutation_scale=18,
            ),
            zorder=2,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ------------------------------------------------------------------
# Example
# ------------------------------------------------------------------

spin_string = (
    "000000000000000000000000000000000000000000000000011111111111111111111111111111111111110000000000000000000000000000000000"
)

plot_ring(spin_string, "ring_state.png")