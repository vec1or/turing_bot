"""Prepare group-split UTF-8 JSONL for local training. Requires Python 3.11+."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile


OASST_URL = ("https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
             "a2c461b87baf5f2c581c20ca75d13d4e7b21c5cb/2023-04-12_oasst_ready.trees.jsonl.gz")
OASST_HASH = "2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7"
SOURCES = {
    "oasst1-ready.jsonl.gz": (OASST_URL, OASST_HASH),
    "rusdracor.zip": ("https://codeload.github.com/dracor-org/rusdracor/zip/cb06b3a211cc36ed0806c9ae85a94ede5da99557",
                      "3485e1d8742b58fe66f20b1dd475334ae64c881f43569ec11df225b005d22cbc"),
    "russian-everyday.json": ("https://huggingface.co/datasets/kukunechka/russian-everyday-dialogues/resolve/"
                              "3d9c43ca85a50e7a32fa7b05e27c0637a710e988/russian_everyday_dialogues.json",
                              "82fc017feeeeda13646914a66f345dd482e7ad9cc858f4fab61ab6d7126be462"),
}
RUS = re.compile(r"[А-Яа-яЁё]")
REJECT = re.compile(r"https?://|www\.|\S+@\S+|\+?\d[\d() -]{8,}\d|```|<[^>]{1,150}>")
ASSISTANT_META = re.compile(
    r"open\s?assistant|openai|chatgpt|чатгпт|языков\w* модел|искусственн\w* интеллект|"
    r"(?:я|ты) (?:[—–-] )?(?:не )?(?:бот|чат.?бот|нейросеть|ассистент|человек|программ\w*)\b|"
    r"мои (?:разработчики|создатели)|(?:(?:при|вы|от|пере)?шли|пришлите|скинь\w*|отправ\w*) "
    r"(?:мне )?(?:ссылк\w*|файл\w*)|(?:я|мне) не (?:могу|позволено) (?:предостав|получить доступ)",
    re.I)
TEI = "{http://www.tei-c.org/ns/1.0}"
AUTHORS = frozenset("afinogenov andreyev artsybashev babel bulgakov chekhov gogol gorky "
                    "leskov mamin naydyonov ostrovsky pisemskij saltykov sukhovo tolstoy turgenev".split())
ARCHAIC = re.compile(r"\b(?:судар\w*|барин\w*|барын\w*|батюшк\w*|матушк\w*|извол\w*|"
                     r"изволь\w*|милостив\w*|кня[зж]\w*|графин\w*|господин\w*|сиятельств\w*|"
                     r"нынче|нынѣ|помилуйте|сие|сей|али|ежели|ежель|извольте|покорнейше|"
                     r"холоп\w*|голубчик\w*|голубушк\w*|почтеннейш\w*)\b|-с\b|[ѣѳі]", re.I)
TAGGED = re.compile(r"[(){}\[\]*<>]|\.{3}|…")


def normalize(text):
    return unicodedata.normalize("NFC", " ".join(text.split()))


def permitted(node):
    if (node.get("lang") != "ru" or node.get("synthetic") or node.get("deleted")
            or node.get("review_result") is not True):
        return False
    labels = node.get("labels") or {}
    if any(labels.get(k, {}).get("value", 0) > .2 for k in
           ("pii", "spam", "lang_mismatch", "not_appropriate", "hate_speech", "sexual_content", "violence", "toxicity")):
        return False
    text = node.get("text", "")
    if not text or REJECT.search(text) or any(c in text for c in "{}[]"):
        return False
    return len(RUS.findall(text)) >= max(2, .75 * sum(c.isalpha() for c in text))


def concise(text):
    text = normalize(text)
    if len(text) > 300:
        # Retain a complete first sentence. Lists, code and ellipses are unsuitable.
        match = re.match(r"^([^.!?]{12,270}[.!?])(?:\s|$)", text)
        if not match or match[1][-2:] == "..":
            return ""
        text = match[1]
    # A sentence boundary in "1." is a list marker, not a complete thought.
    if re.search(r"(?::|[;,…]|\.\.\.|(?:^|[\s:])\d{1,3}[.)]?)\s*$", text):
        return ""
    return text if 3 <= len(text.split()) <= 40 else ""


def oa_rows(archive, stats):
    def walk(node, history):
        if not permitted(node):
            stats["oasst_ineligible_nodes"] += 1
            return
        if ASSISTANT_META.search(node["text"]):
            stats["oasst_service_nodes"] += 1
            return
        utterance = normalize(node["text"])
        if node["role"] == "assistant" and history and history[-1][0] == "prompter":
            response = concise(utterance)
            if response and len(history[-1][1]) <= 350:
                previous = [t for _, t in history[-3:]]
                context = "\n".join(previous)
                if len(context) > 650:
                    context = previous[-1]
                yield context, response
        for child in node.get("replies", []):
            yield from walk(child, history + [(node["role"], utterance)])

    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        for line in stream:
            tree = json.loads(line)
            if tree["prompt"].get("lang") == "ru":
                for context, response in walk(tree["prompt"], []):
                    yield "oasst1", tree["message_tree_id"], context, response


def everyday_rows(path):
    for item in json.loads(path.read_text(encoding="utf-8"))["data"]:
        prompt, response = normalize(item["user"]), concise(item["assistant"])
        if response and not (REJECT.search(prompt + response) or ASSISTANT_META.search(prompt + response)):
            yield "everyday", item["situation"], prompt, response


def spoken_text(node):
    parts = []
    def visit(elem):
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            if child.tag not in (TEI + "stage", TEI + "note", TEI + "speaker"):
                visit(child)
            if child.tail:
                parts.append(child.tail)
    for paragraph in node.findall(TEI + "p"):
        visit(paragraph)
    return normalize(" ".join(parts))


def drama_rows(path, stats):
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            play = Path(name).stem
            if not ("/tei/" in name and name.endswith(".xml") and play.split("-")[0] in AUTHORS):
                continue
            tree = ET.fromstring(archive.read(name))
            license_node = tree.find(f".//{TEI}publicationStmt/{TEI}availability/{TEI}licence")
            if license_node is None or "creativecommons.org/publicdomain/zero/1.0" not in license_node.get("target", ""):
                continue
            # Each immediate sibling block is independent: never cross a scene.
            for block in tree.iter(TEI + "div"):
                history = []
                for speech in block:
                    if speech.tag != TEI + "sp":
                        if speech.tag in (TEI + "div", TEI + "stage"):
                            history.clear()
                        continue
                    who = speech.get("who", "")
                    text = spoken_text(speech)
                    if (not who or " " in who or not 3 <= len(text.split()) <= 30 or
                            len(text) > 210 or len(RUS.findall(text)) < .8 * sum(c.isalpha() for c in text)
                            or ARCHAIC.search(text) or TAGGED.search(text) or REJECT.search(text)
                            or not re.search(r"[.!?][»”]?$", text)):
                        stats["drama_excluded_turns"] += 1
                        history.clear()
                        continue
                    if history and who != history[-1][0]:
                        # About one quarter include the prior exchange. All turns
                        # still belong to this play, scene and consecutive speakers.
                        extended = len(history) == 3 and history[-3][0] == history[-1][0] and history[-2][0] == who
                        use_history = extended and hashlib.sha256(text.encode()).digest()[0] % 4 == 0
                        context = "\n".join(t for _, t in history) if use_history else history[-1][1]
                        yield "rusdracor", play, context, text
                    history = (history + [(who, text)])[-3:]


def prepare(out, cache):
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    collections = [oa_rows(cache / "oasst1-ready.jsonl.gz", stats),
                   drama_rows(cache / "rusdracor.zip", stats), everyday_rows(cache / "russian-everyday.json")]
    assigned, results = {}, {"train": [], "valid": []}
    seen, counts, groups = set(), Counter(), set()
    for collection in collections:
        for source, group, context, response in collection:
            group_id = f"{source}:{group}"
            if group_id not in assigned:
                bucket = int.from_bytes(hashlib.sha256(("arena-v2:" + group_id).encode()).digest()[:4], "big") % 100
                assigned[group_id] = "valid" if bucket < 7 else "train"
            split = assigned[group_id]
            key = tuple(re.sub(r"\W+", " ", normalize(t).casefold().replace("ё","е")).strip() for t in (context, response))
            if key in seen or not all(key):
                continue
            seen.add(key)
            assert 0 < len(response) <= 300 and response == unicodedata.normalize("NFC",response)
            results[split].append(dict(context=context, response=response, source=source, group=group_id))
            counts[f"{source}/{split}"] += 1
            groups.add(group_id)
    for split, records in results.items():
        records.sort(key=lambda rec: hashlib.sha256((rec["group"] + rec["context"] + rec["response"]).encode()).digest())
        path = out / (split + ".jsonl")
        staged = path.with_suffix(".tmp")
        with staged.open("w", encoding="utf-8", newline="\n") as writer:
            for record in records:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        staged.replace(path)
    report = {"groups": len(groups), "counts": dict(sorted(counts.items())), "excluded": dict(stats),
              "sha256": {split: hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest() for split in results}}
    (out / "manifest.json").write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in SOURCES.items():
        path = args.cache / name
        if args.download and not path.exists():
            tmp = path.with_suffix(".part")
            with urlopen(url, timeout=120) as response, tmp.open("wb") as target:
                while chunk := response.read(1024*1024):
                    target.write(chunk)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != checksum:
                raise SystemExit(f"Download SHA-256 mismatch: {name}")
            tmp.replace(path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise SystemExit(f"Cache missing or SHA-256 mismatch: {name}")
    prepare(args.out, args.cache)


if __name__ == "__main__":
    main()
