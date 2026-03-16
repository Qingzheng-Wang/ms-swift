#!/usr/bin/env python3
"""Pretty-print JSON from file or stdin."""
import json
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("json.json")

    if path.exists():
        raw = path.read_text()
    elif len(sys.argv) > 1:
        print(f"File not found: {path}")
        sys.exit(1)
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        print("Usage: pp.py [file]  (default: json.json)")
        print("  Or:  cat data.json | pp.py")
        sys.exit(1)

    raw = raw.strip()
    if not raw:
        print("No input.")
        return
    data = json.loads(raw)
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
