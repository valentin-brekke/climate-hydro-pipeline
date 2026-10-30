#!/usr/bin/env python3
"""Print "<files> <bytes>" for each directory given (a zarr store), so a copy
can be checked against the original: run it on the Mac and on Myriad and
compare. Standard library only; with --mark, writes <store>.done next to
each store whose count and size equal --expect "<files> <bytes>"."""
import argparse
import os
from pathlib import Path


def manifest(path):
    n = size = 0
    for root, _, files in os.walk(path):
        for f in files:
            n += 1
            size += os.path.getsize(os.path.join(root, f))
    return n, size


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stores", nargs="+")
    p.add_argument("--expect", nargs=2, type=int, metavar=("FILES", "BYTES"))
    p.add_argument("--mark", action="store_true")
    args = p.parse_args()
    for s in args.stores:
        n, size = manifest(s)
        line = f"{s} {n} {size}"
        if args.expect:
            ok = (n, size) == tuple(args.expect)
            line += "  match" if ok else f"  MISMATCH (expected {args.expect[0]} {args.expect[1]})"
            if ok and args.mark:
                Path(f"{s.rstrip('/')}.done").write_text(f"{n} {size}\n")
                line += "  -> marked done"
        print(line)


if __name__ == "__main__":
    main()
