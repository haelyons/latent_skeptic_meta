#!/usr/bin/env python3
"""Build stimulus_linewrap.json: public-domain prose materialized at fixed wrap widths.

WHY THIS FILE EXISTS. The seed must embed its stimuli, not describe them: in a
sibling project (mats_r1d1, seed B v1) a stimulus that was only described got
reconstructed by a downstream agent and failed its replication anchor; the
embedded one (stimulus_sitref_primary.json) passed. This script is the one-time
producer; the JSON it writes is the artifact of record. Runners load the JSON
exactly and halt if it is absent. Nobody re-runs this script on a box.

SOURCE. Project Gutenberg eBook #2701 (Moby Dick; Or, The Whale), plain text,
https://www.gutenberg.org/cache/epub/2701/pg2701.txt, public domain. The raw
file's sha256 is recorded in meta.source_sha256. Text between the standard
Gutenberg START/END markers is used; everything else is licensing boilerplate.

CONSTRUCTION, deterministic given the source file:
  1. Split into paragraphs on blank lines; join each paragraph's internal
     newlines with single spaces (this REMOVES Gutenberg's ~70-char hard wrap);
     collapse runs of whitespace to one space.
  2. Normalize to ASCII via an explicit character map (curly quotes, dashes,
     ellipsis, accented vowels); drop any paragraph still containing a
     non-ASCII character. Character counting must be unambiguous: one byte,
     one character, one column.
  3. Keep prose paragraphs of >= 400 characters that are not headings.
  4. Pack consecutive paragraphs into documents of ~DOC_CHARS characters
     (single space between paragraphs, cut at the last word boundary).
  5. First TRAIN_DOCS documents are split "train", the rest "test" - a block
     split in book order, so train and test share no paragraph.
  6. Wrap every document at every width in WIDTHS: greedy word wrap, single
     '\n' between lines, words longer than the width hard-split, no
     hyphenation, no trailing spaces. Every line except a document's last is
     as full as the next word allows; no line exceeds its width.

    python3 make_stimulus_linewrap.py <path-to-pg2701.txt>   # writes JSON here
    python3 make_stimulus_linewrap.py --selftest             # stdlib-only
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

WIDTHS = [20, 40, 60, 80, 100, 120, 140]
N_DOCS = 24
DOC_CHARS = 2400
TRAIN_DOCS = 18
MIN_PARA = 400
OUT = "stimulus_linewrap.json"

ASCII_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "æ": "ae",
    "è": "e", "é": "e", "ê": "e", "à": "a",
    "â": "a", "ç": "c", "ô": "o", "û": "u",
    "ï": "i", "î": "i", " ": " ",
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
        if p.upper() == p:  # headings / TOC
            continue
        out.append(p)
    return out


def pack_docs(paras: list[str]) -> list[str]:
    docs, cur = [], ""
    for p in paras:
        cur = (cur + " " + p).strip() if cur else p
        if len(cur) >= DOC_CHARS:
            cut = cur.rfind(" ", 0, DOC_CHARS)
            docs.append(cur[:cut])
            cur = ""
            if len(docs) == N_DOCS:
                break
    if len(docs) < N_DOCS:
        raise SystemExit(f"only packed {len(docs)} docs; need {N_DOCS}")
    return docs


def wrap(text: str, width: int) -> str:
    lines, line = [], ""
    for word in text.split(" "):
        while len(word) > width:  # hard-split an over-long word
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
    docs = pack_docs(paragraphs(raw))
    records = []
    for i, d in enumerate(docs):
        wrapped = {}
        for w in WIDTHS:
            t = wrap(d, w)
            assert max(len(x) for x in t.split("\n")) <= w
            # hard-split words regain their spelling once whitespace is removed
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
            "generator": "make_stimulus_linewrap.py",
            "wrap_rule": "greedy word wrap, single \\n between lines, words "
                         "longer than width hard-split, no hyphenation",
        },
        "docs": records,
    }


def _selftest() -> int:
    fails = []
    t = wrap("aa bb cc dd", 5)
    if t != "aa bb\ncc dd":
        fails.append(f"greedy wrap: {t!r}")
    t = wrap("abcdefghij xy", 4)
    if t.split("\n")[0:3] != ["abcd", "efgh", "ij"] or max(len(x) for x in t.split("\n")) > 4:
        fails.append(f"hard-split: {t!r}")
    for w in (3, 7, 20):
        src = "the quick brown fox jumps over the lazy dog again and again"
        rt = "".join(wrap(src, w).split())
        if rt != "".join(src.split()):
            fails.append(f"roundtrip w={w}")
        if max(len(x) for x in wrap(src, w).split("\n")) > w:
            fails.append(f"overlong line w={w}")
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
