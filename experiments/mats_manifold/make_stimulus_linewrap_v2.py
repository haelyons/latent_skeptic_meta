#!/usr/bin/env python3
"""Build stimulus_linewrap_v2.json: the seed-D2 stimulus. Same construction as
make_stimulus_linewrap.py (the seed-D1 generator, kept frozen beside this file
as provenance), with three parameter changes and one disjointness step, each
forced by a measured seed-D1 fact:

  1. WIDTHS gains 200 and 280. Seed D1's calibration measured ambient dimension
     1, which makes the theory's predicted ripple spacing equal to the probe
     window at any window; testing invariance across apertures requires more
     than one wide aperture.
  2. 48 documents of ~3,600 characters (was 24 x ~2,400). Seed D1 measured
     adjacent-label cosine of only 0.384: per-label means were noise-dominated
     at 9-39 test samples per label. Wider apertures spread samples thinner, so
     volume rises about 3x.
  3. TRAIN/TEST split 36/12, still a block split in book order.
  4. DISJOINTNESS: this generator first replays seed D1's exact packing
     (24 docs x 2,400 chars) on the same paragraph stream and discards every
     paragraph that packing consumed, so no character of v2 text appeared in
     stimulus_linewrap.json. The v2 anchor leg is therefore a fresh-text
     replication of the seed-D1 curve, not a re-read of the same text.

    python3 make_stimulus_linewrap_v2.py <path-to-pg2701.txt>
    python3 make_stimulus_linewrap_v2.py --selftest
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

WIDTHS = [20, 40, 60, 80, 100, 120, 140, 200, 280]
N_DOCS = 48
DOC_CHARS = 3600
TRAIN_DOCS = 36
MIN_PARA = 400
OUT = "stimulus_linewrap_v2.json"

V1_N_DOCS = 24          # seed D1 packing, replayed only to skip its text
V1_DOC_CHARS = 2400

ASCII_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "æ": "ae",
    "è": "e", "é": "e", "ê": "e", "à": "a",
    "â": "a", "ç": "c", "ô": "o", "û": "u",
    "ï": "i", "î": "i", " ": " ",
}
START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def normalize(s: str) -> str:
    for k, v in ASCII_MAP.items():
        s = s.replace(k, v)
    return s


def paragraphs(raw: str) -> list[str]:
    body = raw
    if START in body:
        body = body.split(START, 1)[1].split("\n", 1)[1]
    if END in body:
        body = body.split(END, 1)[0]
    out = []
    for block in body.split("\n\n"):
        p = " ".join(block.split())
        p = normalize(p)
        if len(p) < MIN_PARA:
            continue
        if not all(ord(c) < 128 for c in p):
            continue
        if p.upper() == p:
            continue
        out.append(p)
    return out


def pack_docs(paras: list[str], n_docs: int, doc_chars: int):
    """Pack consecutive paragraphs into documents. Returns (docs, n_consumed):
    the documents, and how many paragraphs were consumed producing them."""
    docs, cur = [], ""
    consumed = 0
    for i, p in enumerate(paras):
        cur = (cur + " " + p).strip() if cur else p
        if len(cur) >= doc_chars:
            cut = cur.rfind(" ", 0, doc_chars)
            docs.append(cur[:cut])
            cur = ""
            consumed = i + 1
            if len(docs) == n_docs:
                break
    if len(docs) < n_docs:
        raise SystemExit(f"only packed {len(docs)} docs; need {n_docs}")
    return docs, consumed


def wrap(text: str, width: int) -> str:
    lines, line = [], ""
    for word in text.split(" "):
        while len(word) > width:
            if line:
                lines.append(line)
                line = ""
            lines.append(word[:width])
            word = word[width:]
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return "\n".join(lines)


def build(src_path: str) -> dict:
    raw = Path(src_path).read_text(encoding="utf-8")
    paras = paragraphs(raw)
    v1_docs, v1_consumed = pack_docs(paras, V1_N_DOCS, V1_DOC_CHARS)
    v1_sha = hashlib.sha256("\n\n".join(v1_docs).encode()).hexdigest()[:12]
    remaining = paras[v1_consumed:]
    docs, _ = pack_docs(remaining, N_DOCS, DOC_CHARS)
    v2_joined = "\n\n".join(docs)
    for d in v1_docs:                       # disjointness is asserted, not assumed
        probe = d[:200]
        assert probe not in v2_joined, "v2 text overlaps a seed-D1 document"
    records = []
    for i, d in enumerate(docs):
        wrapped = {}
        for w in WIDTHS:
            t = wrap(d, w)
            assert max(len(x) for x in t.split("\n")) <= w
            assert "".join(t.split()) == "".join(d.split())
            wrapped[str(w)] = t
        records.append({
            "id": i,
            "split": "train" if i < TRAIN_DOCS else "test",
            "chars": len(d),
            "sha256_plain": hashlib.sha256(d.encode()).hexdigest()[:12],
            "wrapped": wrapped,
        })
    return {
        "meta": {
            "source": "Project Gutenberg eBook #2701 (Moby Dick), plain text",
            "source_url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
            "source_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "widths": WIDTHS,
            "n_docs": N_DOCS,
            "train_docs": TRAIN_DOCS,
            "doc_chars_target": DOC_CHARS,
            "generator": "make_stimulus_linewrap_v2.py",
            "wrap_rule": "greedy word wrap, single \\n between lines, words "
                         "longer than width hard-split, no hyphenation",
            "disjoint_from": {
                "file": "stimulus_linewrap.json",
                "how": "replayed the seed-D1 packing (24 docs x 2400 chars, "
                       "%d paragraphs consumed) and started after it" % v1_consumed,
                "v1_docs_sha256_joined": v1_sha,
            },
        },
        "docs": records,
    }


def _selftest() -> int:
    fails = []
    t = wrap("aa bb cc dd", 5)
    if t != "aa bb\ncc dd":
        fails.append(f"greedy wrap: {t!r}")
    t = wrap("abcdefghij xy", 4)
    if t.split("\n")[0:3] != ["abcd", "efgh", "ij"]:
        fails.append(f"hard-split: {t!r}")
    for w in (3, 7, 20, 280):
        src = "the quick brown fox jumps over the lazy dog again and again"
        if "".join(wrap(src, w).split()) != "".join(src.split()):
            fails.append(f"roundtrip w={w}")
        if max(len(x) for x in wrap(src, w).split("\n")) > w:
            fails.append(f"overlong line w={w}")
    paras = ["alpha " * 100, "beta " * 100, "gamma " * 100, "delta " * 100]
    paras = [p.strip() for p in paras]
    d1, c1 = pack_docs(paras, 1, 550)
    if c1 != 1 or not d1[0].startswith("alpha"):
        fails.append(f"pack consumed {c1}")
    d2, _ = pack_docs(paras[c1:], 1, 550)
    if d2[0][:5] == d1[0][:5]:
        fails.append("skip did not skip")
    if normalize("“ah” — café") != '"ah" - cafe':
        fails.append("normalize map")
    for f in fails:
        print("FAIL", f)
    print("selftest:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    obj = build(sys.argv[1])
    Path(OUT).write_text(json.dumps(obj, indent=1))
    n_tr = sum(1 for d in obj["docs"] if d["split"] == "train")
    print(f"wrote {OUT}: {len(obj['docs'])} docs ({n_tr} train), widths {WIDTHS}")
