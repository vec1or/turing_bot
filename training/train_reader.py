"""Build aggregate specificity weights from original Russian CC0 Tatoeba sentences."""
import argparse
import bz2
import gzip
import hashlib
import json
from pathlib import Path

from turing_bot.reader import bucket, stem, words


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--valid", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.valid.open(encoding="utf8") as stream:
        excluded = {str(row["sentence_id"]) for line in stream if "sentence_id" in (row := json.loads(line))}
    freq, count = [0] * 16384, 0
    with bz2.open(args.archive, "rt", encoding="utf8") as stream:
        for line in stream:
            fields = line.rstrip().split("\t")
            if len(fields) < 3 or fields[1] != "rus" or fields[0] in excluded:
                continue
            observed = {bucket(stem(t), len(freq)) for t in words(fields[2]) if t.isalpha()}
            if observed:
                count += 1
                for index in observed:
                    freq[index] += 1
    resource = {"format": "arena-reader-idf-1", "documents": count, "document_frequencies": freq,
                "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "reader.json.gz"
    raw = json.dumps(resource, separators=(",", ":")).encode("utf8")
    # GzipFile writes a platform-neutral OS byte, including on Python 3.11.
    with path.open("wb") as handle, gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as archive:
        archive.write(raw)
    print(f"{count} documents, {len(freq)} buckets, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
