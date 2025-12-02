import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_dir = "/exp/mu2e/app/users/hjafree/analysisdir/RefAna/Count/csv_file_cuts"  # directory containing your CSVs

files_info = {
    f"{csv_dir}/all_cuts.csv": "All Cuts",
    f"{csv_dir}/all_but_rmax.csv": "All but Rmax",
    f"{csv_dir}/all_but_d0.csv": "All but D0",
    f"{csv_dir}/all_but_tanDip.csv": "All but tanDip",
    f"{csv_dir}/all_3_removed.csv": "All 3 Removed"
}


# Distinct colors for each cut type
colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']

# Load all datasets and remove "Other"
datasets = {}
for file, label in files_info.items():
    df = pd.read_csv(file)
    df = df[df["Type"].str.lower() != "other"]  # Exclude "Other"
    datasets[label] = df

# Use particle types from one dataset (assumes consistent ordering)
particle_types = datasets["All Cuts"]["Type"].tolist()
n_particles = len(particle_types)
y_pos = np.arange(n_particles)

# Set bar height and spacing
bar_height = 0.12
spacing = 0.02

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, (label, df) in enumerate(datasets.items()):
    offset = (i - len(datasets)/2) * (bar_height + spacing)
    counts = df["Count"].tolist()
    ax.barh(y_pos + offset, counts, height=bar_height, label=label, color=colors[i])

# Y-axis and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(particle_types)
ax.set_xlabel("Event Counts (log scale)")
ax.set_title("MC Truth Yields by Cut Configuration (Excluding 'Other')")
ax.set_xscale("log")

# Move legend outside the plot
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Cut Configurations")

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legend
plt.grid(axis='x', linestyle='--', alpha=0.6, which='both')
plt.show()

