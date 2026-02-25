import boost_histogram as bh
from dask import delayed, compute

@delayed
def process_file(path):
    x, w = read_arrays(path)             # awkward/numpy arrays
    h = bh.Hist(bh.axis.Regular(100, 0, 10, name="x"),
                storage=bh.storage.Weight())
    h.fill(x, weight=w)
    return h

tasks = [process_file(p) for p in paths]
partials = compute(*tasks)              # list of Hist
total = sum(partials)                   # merge into global Hist
