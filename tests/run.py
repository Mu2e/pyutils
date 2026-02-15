#!/usr/bin/env python3
"""Run tests from the run.ipynb as a script.

This script mirrors the notebook `tests/run.ipynb` but exposes a CLI so
you can run selected test groups without executing heavy work by default.

Usage examples:
  python3 tests/run.py --processor
  python3 tests/run.py --reader --importer
  python3 tests/run.py --all
"""

import os
import argparse
import importlib.util

# Print the installed pyutils location (similar to the first notebook cell)
try:
    import pyutils
    print(pyutils.__file__)
except Exception as e:
    print(f"Could not import pyutils: {e}")

# Dynamically import the local tests/pytest.py as a module to avoid
# conflicts with the external `pytest` package.
_this_dir = os.path.dirname(__file__)
_pytest_path = os.path.join(_this_dir, "pytest.py")
_spec = importlib.util.spec_from_file_location("local_pytest", _pytest_path)
_local_pytest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_pytest)

Tester = _local_pytest.Tester


def main():
    parser = argparse.ArgumentParser(description="Run selected pyutils tests (converted from run.ipynb)")
    parser.add_argument("--processor", action="store_true", help="Run processor tests")
    parser.add_argument("--reader", action="store_true", help="Run reader tests")
    parser.add_argument("--importer", action="store_true", help="Run importer tests")
    parser.add_argument("--select", action="store_true", help="Run select tests")
    parser.add_argument("--plot", action="store_true", help="Run plot tests")
    parser.add_argument("--print", dest="print_tests", action="store_true", help="Run print tests")
    parser.add_argument("--vector", action="store_true", help="Run vector tests")
    parser.add_argument("--all", action="store_true", help="Run all test groups")

    args = parser.parse_args()

    tester = Tester()

    if args.all:
        # map all flags to True
        result = tester.run(
            test_reader=True,
            test_processor=True,
            test_importer=True,
            test_select=True,
            test_plot=True,
            test_print=True,
            test_vector=True,
        )
    else:
        result = tester.run(
            test_reader=args.reader,
            test_processor=args.processor,
            test_importer=args.importer,
            test_select=args.select,
            test_plot=args.plot,
            test_print=args.print_tests,
            test_vector=args.vector,
        )

    if result:
        print("All selected tests passed")
        raise SystemExit(0)
    else:
        print("Some tests failed")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
