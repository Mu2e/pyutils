import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")
# compute total_cores (for dask rows)
df["total_cores"] = df.apply(lambda r: (r.n_workers * r.threads_per_worker) if r.mode=="dask" else r.max_workers, axis=1)
baseline = df[(df.mode=="processor") & (df.max_workers==1)]["avg_time_s"].iloc[0]
df["speedup"] = baseline / df["avg_time_s"]

# plot avg time
df.plot.bar(x=df.index, y="avg_time_s")
plt.savefig("avg_time.png")

# plot speedup vs cores
df.plot.scatter(x="total_cores", y="speedup")
plt.savefig("speedup_vs_cores.png")