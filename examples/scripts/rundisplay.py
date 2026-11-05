from pyutils.pydisplay import Display
import argparse

"""
User can run as:

  python rundisplay.py --dataset dts.mu2e.RPCExternal.MDC2020aw.art  --event 200 --run 1202 --subrun 617

By default the EventDisplay config is $MUSE_WORK_DIR/EventDisplay/examples/nominal_MDC2025.fcl,
use --config to choose another one.
"""
def main(args):

  # once you have a feel for the events you might want to display them:
  display = Display(verbosity=args.verbose, config=args.config)
  display.pick_event(str(args.dataset), str(args.run),str(args.subrun),str(args.event))
  display.launch_display(str(args.dataset), str(args.run),str(args.subrun),str(args.event))


if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="mcs data set")
    parser.add_argument("--run", type=int, required=True, help="run number")
    parser.add_argument("--subrun", type=int, required=True,help="sub-run number")
    parser.add_argument("--event", type=int, required=True,help="event number")
    parser.add_argument("--config", type=str, default=None, help="EventDisplay fcl config (default: $MUSE_WORK_DIR/EventDisplay/examples/nominal_MDC2025.fcl)")
    parser.add_argument("--verbose", type=int, default=1, help="verbosity (0: errors only, 1: info, 2: max)")
    args = parser.parse_args()

    # run main function
    main(args)
