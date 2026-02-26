import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("test.csv")

# Convert max_workers to numeric (handles empty strings as NaN)
df["max_workers"] = pd.to_numeric(df["max_workers"], errors="coerce")
df["n_workers"] = pd.to_numeric(df["n_workers"], errors="coerce")
df["threads_per_worker"] = pd.to_numeric(df["threads_per_worker"], errors="coerce")

# compute total_cores (for dask rows)
df["total_cores"] = df.apply(lambda r: (r.n_workers * r.threads_per_worker) if r.mode=="dask" else r.max_workers, axis=1)

# Get baseline: processor with max_workers==1
baseline_rows = df[(df.mode=="processor") & (df.max_workers==1)]
if baseline_rows.empty:
    # Fallback: use the slowest time as baseline
    print("Warning: No processor row with max_workers==1 found. Using slowest time as baseline.")
    baseline = df["avg_time_s"].max()
else:
    baseline = baseline_rows["avg_time_s"].iloc[0]

df["speedup"] = baseline / df["avg_time_s"]

# Create a label for each row to describe the configuration
def row_label(r):
    if r.mode == "processor":
        return f"Processor (mw={int(r.max_workers)})"
    else:
        nw = int(r.n_workers) if pd.notna(r.n_workers) else "?"
        tpw = int(r.threads_per_worker) if pd.notna(r.threads_per_worker) else "?"
        return f"Dask (nw={nw}, tpw={tpw})"

df["label"] = df.apply(row_label, axis=1)

# plot avg time
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df["label"], df["avg_time_s"])
ax.set_ylabel("Average Time (s)")
ax.set_xlabel("Configuration")
ax.set_title("Average Runtime by Configuration")
plt.xticks(rotation=45, ha='right')
fig.tight_layout()
plt.savefig("avg_time.png")
plt.close()

# plot speedup vs cores
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["total_cores"], df["speedup"], s=100)
ax.set_xlabel("Total Cores")
ax.set_ylabel("Speedup (baseline/time)")
ax.set_title("Speedup vs Total Cores")
plt.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("speedup_vs_cores.png")
plt.close()