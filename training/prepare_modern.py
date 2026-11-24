"""Build a spoken/causal subset of Russian Tatoeba and mix with the base corpus."""
import argparse
import bz2
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

from prepare_data import ARCHAIC, ASSISTANT_META, REJECT, RUS, normalize

LOCAL = Path(__file__).resolve().parent
SENTENCE_URL = "https://tatoeba.org/en/sentences/show/"
SAMPLE = LOCAL / "data" / "tatoeba-selected.tsv.gz"
WEEKLY_SHA256 = "338d00cb7badb9f9e6ee64a523db956fdac1cab8da4cacd62cba381ac06e3662"
SELECTED_SHA256 = "f0210c8653d75af7c66545d8a4fdaaf5584d2140cbf1b3a9eb7b6a52f900833c"
QUOTES = " «»„“”\"—–-"
NAMES = re.compile(r"\b(?:том(?:а|у|е|ом)?|м[эе]ри|джон(?:а|у|е|ом)?|бостон\w*|татоэб\w*)\b", re.I)
UNSUITABLE = re.compile(r"\b(?:суицид\w*|самоубий\w*|наркотик\w*|взрывчат\w*|изнасил\w*|порно\w*|застрел\w*|убий\w*|убить|убью|убил\w*|сдох\w*|проститут\w*)\b", re.I)
NARRATION = re.compile(r"^(?:спросил\w*|сказал\w*|ответил\w*|подумал\w*|прошептал\w*|произн[её]с\w*|"
                       r"(?:он|она|я) (?:спросил\w*|сказал\w*|ответил\w*))\b", re.I)
QUESTION_REPLY = re.compile(r'^(.+?\?)[»"”]?\s*(?:[—–-]\s*|[«"„]\s*)(.+)$')
TWO_SENTENCES = re.compile(r"^([^.!?]{5,115}[?!.])\s+([А-ЯЁ][^.!?]{5,125}[.?!])$")
CLAUSES = (
    ("cause", re.compile(r"^([^,;.!?]{5,140}),\s*(?:потому что|так как)\s+([^,;.!?]{5,150})[.!]$", re.I)),
    ("conditional", re.compile(r"^Если\s+([^,;.!?]{5,128}),\s*(?:то\s+)?([^,;.!?]{5,135})[.!]$", re.I)),
    ("time", re.compile(r"^Когда\s+([^,;.!?]{5,120}),\s*([^,;.!?]{5,120})[.!]$", re.I)),
    ("consequence", re.compile(r"^([^,;.!?]{5,125}),\s*(?:поэтому|так что)\s+([^,;.!?]{5,135})[.!]$", re.I)),
)
LIMITS = {"dialogue": 7500, "continuation": 6500, "cause": 4000, "conditional": 4000, "time": 3200, "consequence": 2300}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capitalized(text):
    return text[:1].upper() + text[1:]


def normalized_key(text):
    return " ".join(re.findall(r"\w+", normalize(text).casefold().replace("ё","е")))


def extract(raw):
    text = normalize(raw)
    if not 15 <= len(text) <= 260 or REJECT.search(text) or ASSISTANT_META.search(text) or ARCHAIC.search(text):
        return None
    if UNSUITABLE.search(text) or NAMES.search(text) or len(RUS.findall(text)) < .92 * sum(c.isalpha() for c in text):
        return None
    if re.search(r"[\[\]{}*/@]|\.{3}|…|[!?]{2,}|\d{3,}", text):
        return None
    text = text.strip(QUOTES)
    if "?" in text:
        # Split two speakers only when the question and answer occur in the
        # same contributed example. Never pair unrelated Tatoeba sentences.
        match = QUESTION_REPLY.fullmatch(text)
        if match:
            q,a = (part.strip(QUOTES) for part in match.groups())
            if (2 <= len(q.split()) <= 23 and 2 <= len(a.split()) <= 28
                    and a.endswith((".", "!", "?")) and not NARRATION.search(a)
                    and not re.search(r"[«»\"„“”]", q + a) and "?" not in a[:-1]):
                return "dialogue", q, a
    # Without an explicit break between speakers this is completion of
    # consecutive sentences, not a verified exchange between two people.
    two = TWO_SENTENCES.fullmatch(text)
    if two:
        first, second = two.groups()
        if (2 <= len(first.split()) <= 20 and 2 <= len(second.split()) <= 22
                and not NARRATION.search(second) and not re.search(r"[«»\"„“”—:]", text)):
            return "continuation", first, second
    for kind, pattern in CLAUSES:
        match = pattern.fullmatch(text)
        if not match:
            continue
        first, second = match.groups()
        if not (2 <= len(first.split()) <= 18 and 2 <= len(second.split()) <= 22):
            return None
        if re.search(r"[«»\"„“”—:]", first + second):
            return None
        if kind == "cause":
            return kind, capitalized(first) + ". Почему?", "Потому что " + second + "."
        if kind == "conditional":
            return kind, "А если " + first + "?", capitalized(second) + "."
        if kind == "time":
            return kind, "Когда " + second + "?", "Когда " + first + "."
        return kind, capitalized(first) + ". Что из этого вышло?", capitalized(second) + "."
    return None


def select(raw_path, selected_path):
    candidates = []
    with bz2.open(raw_path, "rt", encoding="utf-8", newline="") as stream:
        for fields in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(fields) != 6 or fields[1] != "rus" or not fields[3].strip() or fields[3] == "\\N":
                continue
            item = extract(fields[2])
            if item:
                source_id, text, author = fields[0], normalize(fields[2]), normalize(fields[3])
                candidates.append((digest(source_id), item[0], source_id, author, text))
    counts, seen = Counter(), set()
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["id","author","original"])
            for _, kind, source_id, author, text in sorted(candidates):
                key = normalized_key(text)
                if counts[kind] == LIMITS[kind] or key in seen:
                    continue
                counts[kind] += 1
                seen.add(key)
                writer.writerow([source_id, author, text])
    return counts


def mix(base, selected, output):
    records = {"train": [], "valid": []}
    stats, seen, credits = Counter(), set(), {}
    # Independent publication groups keep the splits of the prior snapshot.
    # Only a small sample of historical drama is retained in the modern mix.
    for split in records:
        for line in (base / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["source"] == "rusdracor" and (
                    int(digest(line)[:8],16) % 100 > 4 or
                    re.search(r"превосходительств|ваш[ае] (?:светлость|честь)|ваше благородие|любезн\w*|государ\w*|ужели|вельмож\w*|барышн\w*",
                              row["context"] + row["response"], re.I)):
                continue
            key = (normalized_key(row["context"]), normalized_key(row["response"]))
            seen.add(key)
            records[split].append(row)
            stats[split + "/" + row["source"]] += 1
    with gzip.open(selected, "rt", encoding="utf-8", newline="") as stream:
        for src in csv.DictReader(stream, delimiter="\t"):
            source_id = src["id"]
            kind, context, response = extract(src["original"])
            key = normalized_key(context), normalized_key(response)
            if key in seen:
                continue
            seen.add(key)
            # Alternative continuations of an identical question stay together.
            group = "tatoeba:" + digest(key[0])
            split = "valid" if int(digest(group)[:8],16) % 100 < 7 else "train"
            records[split].append(dict(context=context, response=response, source="tatoeba_"+kind,
                                       group=group, sentence_id=source_id, author=src["author"]))
            stats[split + "/tatoeba_" + kind] += 1
            credits[source_id] = src["author"]
    output.mkdir(parents=True, exist_ok=True)
    sums = {}
    for split, rows in records.items():
        rows.sort(key=lambda r: digest(r["group"] + r["context"] + r["response"]))
        path = output / f"{split}.jsonl"
        staged = path.with_suffix(".tmp")
        with staged.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        staged.replace(path)
        sums[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (output / "tatoeba-credits.tsv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.writer(stream, delimiter="\t",lineterminator="\n")
        writer.writerow(["id","author","source"])
        for i in sorted(credits, key=int):
            writer.writerow([i, credits[i], SENTENCE_URL+i])
    manifest = dict(rows=dict(sorted(stats.items())), sha256=sums, tatoeba_sentences=len(credits))
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--selected",type=Path,default=SAMPLE)
    cli.add_argument("--from-weekly-dump",type=Path,help="Rebuild the attributed subset from a Russian detailed TSV.bz2.")
    args = cli.parse_args()
    if args.from_weekly_dump:
        if hashlib.sha256(args.from_weekly_dump.read_bytes()).hexdigest() != WEEKLY_SHA256:
            raise SystemExit("This build uses the 2026-11-21 Tatoeba snapshot; SHA-256 mismatch.")
        print(select(args.from_weekly_dump,args.selected))
    if hashlib.sha256(args.selected.read_bytes()).hexdigest() != SELECTED_SHA256:
        raise SystemExit("Attributed Tatoeba sample SHA-256 mismatch.")
    mix(args.base,args.selected,args.out)
