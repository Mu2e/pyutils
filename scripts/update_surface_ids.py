#! /usr/bin/env python
"""Regenerate pyutils/configs/surface_ids.json from Offline's SurfaceId.hh

The SurfaceId enum in Offline (DataProducts/inc/SurfaceId.hh) is the single source
of truth for surface IDs ("sid") stored in EventNtuple. This script parses that enum
and writes the name -> ID map used by pyselect.Select.

Usage:
    python scripts/update_surface_ids.py                          # fetch from Mu2e/Offline main on GitHub
    python scripts/update_surface_ids.py --offline $MUSE_WORK_DIR/Offline   # local Offline checkout
    python scripts/update_surface_ids.py --header path/to/SurfaceId.hh
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.request

GITHUB_URL = "https://raw.githubusercontent.com/Mu2e/Offline/main/DataProducts/inc/SurfaceId.hh"
HEADER_PATH = "DataProducts/inc/SurfaceId.hh"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "pyutils", "configs", "surface_ids.json")

def parse_surface_ids(text):
    """Parse the SurfaceIdDetail::enum_type enum into a {name: id} dict

    Enumerators either have an explicit value (IPA=90) or take the previous value + 1.
    The result is checked against nSurfaceIds, which Offline maintains by hand.
    """
    match = re.search(r"enum\s+enum_type\s*\{(.*?)\};", text, re.S)
    if match is None:
        raise ValueError("Could not find 'enum enum_type { ... };' in header")
    # Strip comments
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", match.group(1), flags=re.S)

    surface_ids = {}
    next_value = 0
    for token in (t.strip() for t in body.split(",")):
        if not token:
            continue
        name, _, value = (part.strip() for part in token.partition("="))
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"Unexpected enumerator '{token}'")
        if value:
            next_value = int(value)
        surface_ids[name] = next_value
        next_value += 1

    # nSurfaceIds counts 'unknown' as well
    n_match = re.search(r"nSurfaceIds\s*=\s*(\d+)", text)
    if n_match and int(n_match.group(1)) != len(surface_ids):
        raise ValueError(f"Parsed {len(surface_ids)} enumerators but nSurfaceIds = {n_match.group(1)}")

    surface_ids.pop("unknown", None)
    return surface_ids

def _git_commit(path):
    """Return the git commit of the repo containing path, if any"""
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--offline", help="Path to a local Offline checkout")
    source.add_argument("--header", help="Path to SurfaceId.hh")
    parser.add_argument("--out", default=OUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    if args.header or args.offline:
        header = args.header or os.path.join(args.offline, HEADER_PATH)
        with open(header) as f:
            text = f.read()
        commit = _git_commit(os.path.dirname(os.path.abspath(header)))
        source_desc = os.path.abspath(header) + (f" (commit {commit})" if commit else "")
    else:
        with urllib.request.urlopen(GITHUB_URL, timeout=30) as response:
            text = response.read().decode()
        source_desc = GITHUB_URL

    surface_ids = parse_surface_ids(text)

    # Report changes relative to the existing file
    old_ids = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            old_ids = json.load(f).get("surface_ids", {})
    added = sorted(set(surface_ids) - set(old_ids))
    removed = sorted(set(old_ids) - set(surface_ids))
    changed = sorted(k for k in set(surface_ids) & set(old_ids) if surface_ids[k] != old_ids[k])

    output = {
        "source": source_desc,
        "generated": datetime.date.today().isoformat(),
        "surface_ids": dict(sorted(surface_ids.items(), key=lambda item: item[1])),
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(surface_ids)} surface IDs to {os.path.normpath(args.out)}")
    print(f"\tSource: {source_desc}")
    if added:
        print(f"\tAdded: {added}")
    if removed:
        print(f"\tRemoved: {removed}")
    if changed:
        print(f"\tChanged value (check before committing!): {changed}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
