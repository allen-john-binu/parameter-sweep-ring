import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Hardcode your results CSV here
# ------------------------------------------------------------------
csv_path = "ztParameterStudySample3/parameter_study_results.csv"

# Read only the first row
df = pd.read_csv(csv_path)
row = df.iloc[0]

# Extract all bump-angle columns
angle_columns = [
    c for c in df.columns
    if c.endswith("_bumpangle")
]

bump_angles = row[angle_columns].astype(float).to_numpy()

# DOA timestep (1,2,3,...)
timesteps = range(1, len(bump_angles) + 1)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.plot(
    timesteps,
    bump_angles,
    linewidth=1.5,
)

plt.xlabel("DOA timestep")
plt.ylabel("Average bump angle (degrees)")
plt.title("Average bump angle vs DOA timestep")

plt.ylim(-180, 180)
plt.xlim(1, len(bump_angles))

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()