#! /usr/bin/env python
"""
Calorimeter event displays with pyutils.pycalo

Draws the crystals with digis (yellow) and reco digis (cyan) for selected events,
one PNG per event and disk. The ntuple must contain the calodigis and calorecodigis
branches (EventNtuple with FillCaloDigis and FillCaloRecoDigis set to true).

Usage:
    python calo_event_display.py --file ntuple.root                 # busiest event in the first 500
    python calo_event_display.py --file ntuple.root --events 3 17 --disks 0 1 --pipes
    python calo_event_display.py --file ntuple.root --n-read 2000 --out-dir displays
"""
import argparse
import os

import awkward as ak
import matplotlib
matplotlib.use("Agg") # write files, no windows
import matplotlib.pyplot as plt
import numpy as np

from pyutils.pycalo import CaloVisualization
from pyutils.pylogger import Logger
from pyutils.pyread import Reader

BRANCHES = ["calodigis.SiPMID_", "calodigis.diskID_", "calorecodigis.caloDigiIdx_"]

def main(args):
    logger = Logger(print_prefix="[calo_event_display]", verbosity=args.verbosity)

    # Read only the events we need, the calo branches can be large
    reader = Reader(verbosity=args.verbosity)
    with reader.read_file(args.file) as file:
        data = file[args.tree_path].arrays(BRANCHES, entry_stop=args.n_read, library="ak")
    logger.log(f"Read {len(data)} events from {args.file}", "info")

    # Default to the event with the most digis
    events = args.events
    if not events:
        events = [int(ak.argmax(ak.num(data["calodigis.SiPMID_"])))]
        logger.log(f"No events given, using the busiest event: {events[0]}", "info")

    calo_vis = CaloVisualization(verbosity=args.verbosity)
    if not calo_vis.load_crystal_map():
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    for evt_idx in events:
        if evt_idx >= len(data):
            logger.log(f"Event {evt_idx} not read (only {len(data)} events, increase --n-read)", "warning")
            continue

        # Default to the disks with digis in this event
        disks = args.disks or sorted(np.unique(ak.to_numpy(data["calodigis.diskID_"][evt_idx])).tolist())
        for disk in disks:
            out_path = os.path.join(args.out_dir, f"calo_event{evt_idx}_disk{disk}.png")
            fig, ax = calo_vis.plot_event_display(
                data=data,
                evt_idx=evt_idx,
                disk=disk,
                show_pipes=args.pipes,
                output_file=out_path
            )
            if fig is not None:
                plt.close(fig)
                logger.log(f"Wrote {out_path}", "success")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, help="EventNtuple file with calodigis and calorecodigis branches")
    parser.add_argument("--events", type=int, nargs="+", help="Event indices to display (default: busiest event)")
    parser.add_argument("--disks", type=int, nargs="+", choices=[0, 1], help="Disks to display (default: disks with digis)")
    parser.add_argument("--n-read", type=int, default=500, help="Number of events to read from the start of the file (default: 500)")
    parser.add_argument("--pipes", action="store_true", help="Overlay the calibration source pipes")
    parser.add_argument("--out-dir", default=".", help="Output directory for PNG files (default: current directory)")
    parser.add_argument("--tree-path", default="EventNtuple/ntuple", help="Path to the ntuple in the file")
    parser.add_argument("--verbosity", type=int, default=1, help="Printout level (0: errors only, 1: info, 2: max)")
    raise SystemExit(main(parser.parse_args()))
