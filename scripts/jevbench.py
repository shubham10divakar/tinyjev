"""JevBench-mini tools: validate | agreement | freeze (see jevbench/README.md)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevcore.data import jevbench  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["validate", "agreement", "freeze"])
    ap.add_argument("path", nargs="?", default="jevbench/items.jsonl")
    a = ap.parse_args(argv)
    items = jevbench.read(a.path)
    if a.command == "validate":
        rep = jevbench.check(items)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 1 if rep["problems"] else 0
    if a.command == "agreement":
        k, n = jevbench.cohen_kappa(items)
        print(f"Cohen's kappa = {k:.3f} on {n} doubly-annotated items" if n else "no label_2 yet")
        return 0
    print(json.dumps(jevbench.freeze(a.path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
