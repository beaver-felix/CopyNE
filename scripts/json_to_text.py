#!/usr/bin/env python3
import sys
from pathlib import Path

# Package priority: orjson >> json
try:
    import orjson

    def parse_line(line: bytes) -> dict:
        return orjson.loads(line)
except ImportError as err:
    import json

    def parse_line(line: bytes) -> dict:
        return json.loads(line.decode("utf-8"))


def main() -> None:
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python json_to_text.py <input.json> <output.text>\n")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    text_path = Path(sys.argv[2])

    if not json_path.exists():
        sys.stderr.write(f"Error: {json_path} does not exist\n")
        sys.exit(1)

    text_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "rb") as fin, open(text_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = parse_line(line)
            fout.write(f"{item['key']} {item['txt']}\n")


if __name__ == "__main__":
    main()
