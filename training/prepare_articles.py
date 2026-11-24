"""Prepare attributed paragraphs from pinned revisions of Russian Wikipedia."""
import argparse
from collections import Counter
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATA = Path(__file__).resolve().parent / "data"
API = "https://ru.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0"
CAUSAL = re.compile(
    r"потому что|вследствие|из-за|благодаря|поэтому|в результате|"
    r"обусловлен|приводит к|приводят к|связано с|объясняется|"
    r"влияни[ея]|формиру[ею]тся|образу[ею]тся|причин[а-я]{0,3}\b", re.I)
END_SECTIONS = re.compile(r"^(?:см\.? также|примечания|источники|литература|ссылки|библиография|галерея|дополнительная литература|внешние ссылки)", re.I)
IGNORE_SECTIONS = re.compile(r"в (?:искусстве|литературе|популярной культуре)|наград[ыа]|известные люди|символик|нумизматик|филатели|кинематографе|фотогалере|цитат",re.I)
UNWANTED = re.compile(r"\b(?:наркотик|взрывчат|изнасил|порнограф|дозировк|передозиров|расстрел|террорист|самоубийств|противотанков|огнестрельн|вооружени|оружи[еяю]|боеприпас)", re.I)
SENTENCE_END = re.compile(r"[.!?…][»”)]?$")


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def query(params):
    params = {"action": "query", "format": "json", "formatversion": "2", **params}
    request = Request(API + "?" + urlencode(params),
                      headers={"User-Agent": "TuringArenaCorpusBuilder/2.0 (educational offline dataset)",
                               "Accept-Encoding": "gzip"})
    for attempt in range(12):
        try:
            with urlopen(request, timeout=90) as stream:
                raw = stream.read()
                if stream.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            result = json.loads(raw)
            if "error" in result:
                raise ValueError(result["error"])
            time.sleep(4.5)
            return result["query"]
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 11:
                raise
            if isinstance(error, HTTPError) and error.code == 429:
                retry_after = error.headers.get("Retry-After", "10")
                delay = int(retry_after) if retry_after.isdigit() else 10
                time.sleep(max(6, delay + 2))
            elif isinstance(error, HTTPError) and error.code < 500:
                raise
            else:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("No response received")


def discover(topics):
    requested = {}
    for line in topics.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            domain, _, titles = line.partition(":")
            for title in titles.split("|"):
                requested.setdefault(title.strip(), domain)
    pages, missing = {}, []
    for group in batches(list(requested), 32):
        result = query({"titles": "|".join(group), "prop": "revisions",
                        "rvprop": "ids|timestamp", "redirects": "1"})
        normalized = {p["from"]: p["to"] for p in result.get("normalized", [])}
        redirects = {p["from"]: p["to"] for p in result.get("redirects", [])}
        hits = {p["title"]: p for p in result["pages"]}
        for item in group:
            target = normalized.get(item, item)
            target = redirects.get(target, target)
            page = hits.get(target)
            if not page or "revisions" not in page or page["ns"] != 0:
                missing.append(item)
                continue
            rev = page["revisions"][0]
            pages.setdefault(page["pageid"], {"title": page["title"], "pageid": page["pageid"],
                 "revid": rev["revid"], "timestamp": rev["timestamp"],
                 "domain": requested[item]})
    return {"project": "ru.wikipedia.org", "license": LICENSE, "missing_titles": missing,
            "articles": sorted(pages.values(), key=lambda p: p["pageid"])}


def revision_texts(pages, cache, download):
    cache.mkdir(parents=True, exist_ok=True)
    missing = [p for p in pages if not (cache / (str(p["revid"]) + ".txt")).exists()]
    if missing and not download:
        raise ValueError("Some pinned source revisions are absent from the cache. Use --download.")
    for group in batches(missing, 24):
        result = query({"revids": "|".join(str(p["revid"]) for p in group),
                        "prop": "revisions", "rvslots": "main", "rvprop": "ids|content"})
        for page in result["pages"]:
            for rev in page.get("revisions", []):
                content = rev["slots"]["main"]["content"]
                dest = cache / (str(rev["revid"]) + ".txt")
                staged = dest.with_suffix(".tmp")
                staged.write_text(content, encoding="utf-8", newline="\n")
                staged.replace(dest)
    for page in pages:
        path = cache / (str(page["revid"]) + ".txt")
        raw = path.read_bytes()
        if page.get("sha256") and page["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"Different source content for revision {page['revid']}")
        page["sha256"] = hashlib.sha256(raw).hexdigest()
        yield page, raw.decode("utf-8")


def plain_template(match):
    parts = [s.strip() for s in match[0][2:-2].split("|")]
    name, args = parts[0].casefold(), parts[1:]
    if re.fullmatch(r"(?:м|км|дм|см|мм)[23]",name) and not args:
        return name[:-1] + ("²" if name[-1] == "2" else "³")
    if re.fullmatch(r"lang-[a-z-]{2,12}2?",name) and args:
        return args[0]
    if name == "lang" and len(args) >= 2:
        return args[1]
    if name in {"nobr","nowrap","число","num"} and len(args) == 1:
        return args[0]
    if name in {"sfn","sfnm","harvnb","примечание","проверено","нет аи","уточнить"}:
        return ""
    # A discarded unknown template can remove a numeral, a formula, a
    # physical unit or a name. Mark the affected paragraph unusable.
    return "¤"


def flatten_templates(text):
    pattern = re.compile(r"\{\{[^{}]*\}\}")
    while True:
        simpler, count = pattern.subn(plain_template, text)
        if not count:
            return text
        text = simpler


def paragraphs(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<(ref|gallery|math|chem|syntaxhighlight|source|timeline|score|table|blockquote)\b[^>]*/\s*>", "", text, flags=re.I)
    text = re.sub(r"<(math|chem)\b[^>]*>.*?</\1\s*>", "¤", text, flags=re.I|re.S)
    text = re.sub(r"<(ref|gallery|syntaxhighlight|source|timeline|score|table|blockquote)\b[^>]*>.*?</\1\s*>", " ", text, flags=re.I|re.S)
    text = flatten_templates(text)
    text = re.sub(r"^\s*\{\|.*?^\s*\|\}\s*$", "\n", text, flags=re.S|re.M)
    section, ignored_level, seen = "", None, set()
    for order, raw in enumerate(text.splitlines()):
        line = raw.strip()
        head = re.fullmatch(r"(={2,6})\s*(.*?)\s*\1", line)
        if head:
            level, label = len(head[1]), head[2].strip()
            if ignored_level is not None and level > ignored_level:
                continue
            ignored_level = level if END_SECTIONS.search(label) or IGNORE_SECTIONS.search(label) else None
            section = re.sub(r"[\[\]']", "", label)
            continue
        if ignored_level or not line or line[0] in "|!#*:;=<>{}[]-–—":
            # A normal paragraph may open with an internal link.
            if ignored_level or not line.startswith("[[") or re.match(r"\[\[(?:файл|категория|file|изображение):",line,re.I):
                continue
        line = re.sub(r"\[\[(?:файл|категория|file|изображение):.*?\]\]", "", line, flags=re.I)
        line = re.sub(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]", lambda m: m[2] or m[1], line)
        line = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", line)
        line = re.sub(r"<[^>]+>|'{2,5}", "", line)
        line = html.unescape(line).replace("\u0301", "").replace("\u00ad", "")
        line = unicodedata.normalize("NFC", " ".join(line.split()))
        line = re.sub(r"\(\s*\)", "", line)
        line = re.sub(r"\s+([.,;:!?])", r"\1", line)
        line = " ".join(line.split())
        if (not 55 <= len(line) <= 3600 or not SENTENCE_END.search(line) or line in seen or
            not re.match(r"[А-ЯЁ«A-Z\d]", line) or re.search(r"[¤\[\]{}|<>]|https?://", line) or
            UNWANTED.search(line) or line.count("(") != line.count(")") or
            len(re.findall(r"[а-яё]{2,}", line.casefold())) < 8):
            continue
        seen.add(line)
        yield order, section, line


def noteworthy(rows):
    # Cover the lead, explanations, and subject sections in article order.
    # Equal priority within each class preserves original document order.
    chosen = {row[0] for row in rows[:3]}
    chosen.update(row[0] for row in [r for r in rows if CAUSAL.search(r[2])][:26])
    section_count = Counter()
    for order, section, line in rows:
        if section_count[section] < 2 and len(chosen) < 55:
            chosen.add(order)
            section_count[section] += 1
    for order, _, _ in rows:
        if len(chosen) >= 55:
            break
        chosen.add(order)
    return [row for row in rows if row[0] in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=Path, default=DATA / "article-topics.txt")
    parser.add_argument("--manifest", type=Path, default=DATA / "article-revisions.json")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=DATA / "articles.jsonl.gz")
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned revisions from Wikipedia")
    parser.add_argument("--discover", action="store_true", help="Create a NEW snapshot manifest from the current topics")
    args = parser.parse_args()
    if args.discover:
        if not args.download or args.manifest.exists():
            parser.error("--discover requires --download and a new manifest path")
        manifest = discover(args.topics)
        args.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n", encoding="utf-8", newline="\n")
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_suffix(".tmp")
    with staged.open("wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0) as output:
        for page, text in revision_texts(manifest["articles"], args.cache, args.download):
            selected = noteworthy(list(paragraphs(text)))
            counts["articles"] += bool(selected)
            for order, section, para in selected:
                record = {k:page[k] for k in ("title", "pageid", "revid", "domain")}
                record.update(section=section, text=para, url=f"https://ru.wikipedia.org/w/index.php?oldid={page['revid']}",
                              license=LICENSE)
                output.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                counts["paragraphs"] += 1
                counts["causal_marker"] += bool(CAUSAL.search(para))
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    if manifest.get("artifact_sha256") and manifest["artifact_sha256"] != digest:
        staged.unlink()
        raise ValueError("Output differs from the pinned artifact")
    staged.replace(args.out)
    manifest.update(counts=dict(counts),artifact_sha256=digest)
    args.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"sha256":digest,**counts},ensure_ascii=False))


if __name__ == "__main__":
    main()
