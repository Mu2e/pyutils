from pyutils.pyprocess import Processor
from pyutils.pydisplay import Display
from pyutils.pyprint import Print
import argparse

"""
User can run as:

  python rundisplay.py --dataset dts.mu2e.RPCExternal.MDC2020aw.art  --event 200 --run 1202 --subrun 617

"""
def main(args):

  # once you have a feel for the events you might want to display them:
  display = Display(verbosity=2)
  display.pick_event(str(args.dataset), str(args.run),str(args.subrun),str(args.event))
  display.launch_display(str(args.dataset), str(args.run),str(args.subrun),str(args.event))


if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="mcs data set")
    parser.add_argument("--run", type=int, required=True, help="run number")
    parser.add_argument("--subrun", type=int, required=True,help="sub-run number")
    parser.add_argument("--event", type=int, required=True,help="event number")
    parser.add_argument("--verbose", default=1, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()

    # run main function
    main(args)

