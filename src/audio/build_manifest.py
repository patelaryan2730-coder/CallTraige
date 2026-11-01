"""
Build Training Manifest from ASVspoof Protocol Files
--------------------------------------------------------
Converts an ASVspoof protocol/key file into the simple filepath,label CSV
that train.py expects.


Usage:
    python build_manifest.py \
        --protocol path/to/keys/CM/trial_metadata.txt \
        --audio-dir path/to/flac \
        --audio-dir path/to/another/flac \
        --audio-ext .flac \
        --out ../../data/train_manifest.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter

COLUMN_FILENAME = 1
COLUMN_LABEL = 5

LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}

PROGRESS_EVERY = 50_000


def index_audio_dirs(audio_dirs: list, audio_ext: str) -> dict:
    """
    Scans every --audio-dir ONCE and builds {filename_without_ext: full_path}.
    This is the key speedup - os.scandir() on a folder is one fast bulk
    operation, versus checking os.path.isfile() one file at a time for
    every single protocol line.
    """
    index = {}
    for d in audio_dirs:
        print(f"Indexing {d} ...")
        start = time.time()
        count = 0
        if not os.path.isdir(d):
            print(f"  WARNING: directory does not exist, skipping: {d}")
            continue
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(audio_ext):
                    key = entry.name[: -len(audio_ext)]
                    index[key] = entry.path
                    count += 1
        print(f"  Found {count} files in {time.time() - start:.1f}s")
    print(f"Total unique files indexed: {len(index)}\n")
    return index


def build_manifest(protocol_path: str, audio_dirs: list, audio_ext: str, out_path: str):
    file_index = index_audio_dirs(audio_dirs, audio_ext)

    rows = []
    skipped = 0
    missing_examples = []
    unparseable_examples = []
    field_count_distribution = Counter()

    total_lines = sum(1 for _ in open(protocol_path, "r"))
    print(f"Protocol file has {total_lines} lines. Starting...\n")

    start = time.time()
    with open(protocol_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            if line_num % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                rate = line_num / elapsed if elapsed > 0 else 0
                remaining = (total_lines - line_num) / rate if rate > 0 else 0
                print(
                    f"  ... {line_num}/{total_lines} lines "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                )
                sys.stdout.flush()

            raw_line = line.rstrip("\n")
            parts = raw_line.strip().split()
            field_count_distribution[len(parts)] += 1

            if len(parts) <= max(COLUMN_FILENAME, COLUMN_LABEL):
                skipped += 1
                if len(unparseable_examples) < 8:
                    unparseable_examples.append((line_num, "too few fields", raw_line))
                continue

            filename = parts[COLUMN_FILENAME]
            label_str = parts[COLUMN_LABEL].lower()

            if label_str not in LABEL_MAP:
                skipped += 1
                if len(unparseable_examples) < 8:
                    unparseable_examples.append(
                        (line_num, f"field[{COLUMN_LABEL}]='{parts[COLUMN_LABEL]}' not bonafide/spoof", raw_line)
                    )
                continue

            filepath = file_index.get(filename)
            if filepath is None:
                skipped += 1
                if len(missing_examples) < 5:
                    missing_examples.append(filename)
                continue

            rows.append({"filepath": filepath, "label": LABEL_MAP[label_str]})

    print(f"\nDone processing in {time.time() - start:.1f}s total.\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Skipped {skipped} lines (missing files or unparseable rows)")

    if missing_examples:
        print(f"\nExample filenames that could not be found in any --audio-dir: {missing_examples}")

    if skipped > len(rows) * 0.1 and len(rows) > 0:
        print(
            "\nWARNING: more than 10% of lines were skipped.\n"
            "\n--- DIAGNOSTIC: field count distribution across ALL lines ---"
        )
        for count, freq in sorted(field_count_distribution.items()):
            print(f"  {count} fields: {freq} lines")

        if unparseable_examples:
            print("\n--- DIAGNOSTIC: example unparseable RAW lines (line_num, reason, content) ---")
            for line_num, reason, raw in unparseable_examples:
                print(f"  Line {line_num} ({reason}):")
                print(f"    {raw}")

    if rows:
        bonafide_count = sum(1 for r in rows if r["label"] == 0)
        spoof_count = sum(1 for r in rows if r["label"] == 1)
        print(f"\nClass balance: {bonafide_count} bona fide, {spoof_count} spoofed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio-dir", required=True, action="append")
    parser.add_argument("--audio-ext", default=".flac")
    parser.add_argument("--out", default="../../data/train_manifest.csv")
    args = parser.parse_args()

    build_manifest(args.protocol, args.audio_dir, args.audio_ext, args.out)