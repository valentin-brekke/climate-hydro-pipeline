#!/usr/bin/env python3
"""Check a copy of zarr stores against the original: print "<store> <files>
<bytes>" for each, run on the Mac and on Myriad, and compare. Standard
library only.

  store_manifest.py STORE ...                      print the manifest
  store_manifest.py STORE --expect FILES BYTES     compare one store
  store_manifest.py --expect-file MANIFEST         compare every store listed
                                                   (paths relative to the cwd)
With --mark, writes <store>.done next to each store that matches; a store
that is already marked is left alone.
"""
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


def compare(store, expected, mark):
    done = Path(f"{store.rstrip('/')}.done")
    if mark and done.exists():
        return f"{store} already marked done"
    if not Path(store).exists():
        return f"{store} missing"
    n, size = manifest(store)
    ok = (n, size) == tuple(expected)
    line = f"{store} {n} {size}  " + ("match" if ok else f"MISMATCH (expected {expected[0]} {expected[1]})")
    if ok and mark:
        done.write_text(f"{n} {size}\n")
        line += "  -> marked done"
    return line


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stores", nargs="*")
    p.add_argument("--expect", nargs=2, type=int, metavar=("FILES", "BYTES"))
    p.add_argument("--expect-file")
    p.add_argument("--mark", action="store_true")
    args = p.parse_args()
    if args.expect_file:
        for line in Path(args.expect_file).read_text().splitlines():
            if line.strip():
                store, n, size = line.split()
                print(compare(store, (int(n), int(size)), args.mark))
        return
    for s in args.stores:
        if args.expect:
            print(compare(s, args.expect, args.mark))
        else:
            n, size = manifest(s)
            print(f"{s} {n} {size}")


if __name__ == "__main__":
    main()
