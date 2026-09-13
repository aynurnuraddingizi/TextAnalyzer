"""One-time build script: collapses the raw ELELex.tsv (CEFRLex project,
Spanish sub-lexicon — see README.md in this folder for the source and
license) into cefr_es.json at the project root, the compact bundled
resource text_analyzer.py actually reads at runtime — same
source/derived-resource split as build_cefr_json.py's own cefr_en.json.

Run it again only if the source TSV changes:
    python cefr_data/build_cefr_es_json.py /path/to/ELELex.tsv

Why this needs its own build script rather than reusing
build_cefr_json.py's: ELELex's raw shape is fundamentally different from
CEFR-J/Octanove's. Those two CSVs each already have ONE level column per
row ("this word IS level X"). ELELex instead gives, per (word, part-of-
speech) row, that word's own normalized frequency AND document count at
EACH of five levels (a1/a2/b1/b2/c1 — there is no c2 tier in this data
at all, a real, honestly-documented gap, not an oversight) — collapsing
that distribution down to the single level this app's cefr_level() wants
is a real judgment call, not just reformatting, so it's written out and
justified here in full.

Method: a word's level is the EARLIEST (lowest) of a1/a2/b1/b2/c1 at
which it has genuine, non-marginal attestation — "genuine" meaning it
appeared in at least MIN_DOCS separate documents at that level, not
just frequency > 0 (which a single repeated mention, a proper noun
collision, or a corpus/tagging artifact can produce on its own). This
mirrors the everyday meaning "CEFR level" already carries elsewhere in
this app and in the source literature (CEFR-J, Kelly): the level a
learner would first plausibly need to know this word by — not the
level it's used MOST at (a "peak" definition), and not a level implied
by averaging its frequency across all five bands (a "centre of mass"
definition) — CEFRLex's own documentation confirms neither of those
two alternatives is the resource's own prescribed method; deriving a
single level from this data is left to whoever consumes it. Confirmed
directly against real words before choosing MIN_DOCS = 2: at
MIN_DOCS = 1, "casa" (house) correctly resolves A1, but so does a
same-spelled, differently-tagged near-noise row backed by a single
document; at MIN_DOCS = 2, that noise row is dropped while "casa"
(genuinely attested across hundreds of A1 documents) is unaffected —
2 was the smallest threshold that filtered the single-document noise
observed directly in this file without losing well-attested words.

Word+POS pairs seen more than once after tag-mapping (see
_POS_FOR_TAG() below — every noun tag variant, for instance, collapses
to the same "noun" bucket) keep the EASIEST level, same reasoning and
same rule as build_cefr_json.py's own merge step: this data tells a
reader how hard a word is, and a word with any well-attested easy
sense shouldn't be flagged harder than that sense.

Skipped entirely: multi-word entries (ELELex joins these with
underscores, e.g. "sin_embargo") — dead data for this app's single-
token tokenizer, exactly the same reasoning build_cefr_json.py already
documents for the English data's own multi-word headwords.
"""
import csv
import json
import os
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(os.path.dirname(FOLDER), "cefr_es.json")

LEVELS = ("a1", "a2", "b1", "b2", "c1")
LEVEL_LABELS = {"a1": "A1", "a2": "A2", "b1": "B1", "b2": "B2", "c1": "C1"}
LEVEL_RANK = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4}
MIN_DOCS = 2

# FreeLing/EAGLES tagset (the tagger ELELex's own source corpus was
# annotated with) — only the FIRST letter is used, which is always the
# broad category regardless of the (often malformed, in a couple dozen
# rows out of 14,290) rest of the tag string: N=noun (common or proper
# — WordNet-derived POS elsewhere in this app doesn't distinguish those
# either, so neither does this), V=verb, A=adjective, R=adverb.
# Everything else (D=determiner, C=conjunction, P=pronoun, S=
# preposition, I=interjection, Z=number, F=punctuation, or a tag this
# loosely doesn't even recognize) maps to "" — no WordNet-style POS
# category applies, same fallback build_cefr_json.py already uses for
# an English row with no POS recorded.
_POS_FOR_TAG_LETTER = {"N": "noun", "V": "verb", "A": "adjective", "R": "adverb"}


def _pos_for_tag(tag):
    tag = (tag or "").strip()
    if not tag:
        return ""
    return _POS_FOR_TAG_LETTER.get(tag[0].upper(), "")


def _first_qualifying_level(row):
    for level in LEVELS:
        try:
            nb_doc = float(row.get(f"nb_doc@{level}") or 0)
        except ValueError:
            nb_doc = 0
        if nb_doc >= MIN_DOCS:
            return LEVEL_LABELS[level]
    return None


def main(raw_path):
    words = {}
    total_rows = 0
    kept_rows = 0
    with open(raw_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            word = (row.get("word") or "").strip().lower()
            if not word or "_" in word:
                continue
            level = _first_qualifying_level(row)
            if level is None:
                continue
            pos = _pos_for_tag(row.get("tag"))
            by_pos = words.setdefault(word, {})
            existing = by_pos.get(pos)
            if existing is None or LEVEL_RANK[level] < LEVEL_RANK[existing]:
                by_pos[pos] = level
            kept_rows += 1

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=0, sort_keys=True)

    print(f"Processed {total_rows} raw rows, kept {kept_rows} (MIN_DOCS={MIN_DOCS}).")
    print(f"Wrote {len(words)} words to {OUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_cefr_es_json.py /path/to/ELELex.tsv")
        sys.exit(1)
    main(sys.argv[1])
