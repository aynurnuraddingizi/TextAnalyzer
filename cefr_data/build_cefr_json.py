"""One-time build script: combines the two raw CEFR-J / Octanove source
CSVs in this folder into cefr_en.json at the project root, the compact
bundled resource text_analyzer.py actually reads at runtime (same split
as the Russian dictionary's build script mentioned in text_analyzer.py:
source/build material lives in the project, only the derived resource
ships with the app).

Run it again only if the source CSVs change:
    python cefr_data/build_cefr_json.py

Sources (see README.md in this folder for full license text):
  - cefrj-vocabulary-profile-1.5.csv   (CEFR-J, A1-B2) - Tono Laboratory,
    Tokyo University of Foreign Studies. Free for research/commercial
    use with citation.
  - octanove-vocabulary-profile-c1c2-1.0.csv (C1/C2) - Octanove Labs,
    under CC BY-SA 4.0. cefr_en.json, being a derivative that includes
    this data, is itself distributed under CC BY-SA 4.0 - see README.md.

Output shape: {"word": {"pos": "A1", ...}}, where "pos" is either one of
the CSVs' own part-of-speech strings (lowercase, e.g. "noun"/"verb"/
"preposition") or "" for a row with no POS recorded at all. A word with
multiple senses at different levels keeps one level per POS it actually
appears under; within one (word, POS) pair seen more than once (a few
dozen cases in the Octanove file, distinguished only by a free-text
sense note this build discards) the EASIEST level wins, since this data
is used to tell a reader how hard a word is, and a word that has any
common easy sense shouldn't be flagged as harder than that.
"""
import csv
import json
import os

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(os.path.dirname(FOLDER), "cefr_en.json")

# Order matters only for documentation, not merging: CEFR-J covers A1-B2
# and Octanove covers C1/C2, so no (word, pos) pair is ever supplied by
# both files.
SOURCES = [
    "cefrj-vocabulary-profile-1.5.csv",
    "octanove-vocabulary-profile-c1c2-1.0.csv",
]

LEVEL_RANK = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

# Confirmed typo in the source data (remonstrate,vern,C2 - "remonstrate"
# is a verb): fixed here rather than left to silently fail every lookup.
POS_FIXES = {"vern": "verb"}


def load_rows(filename):
    path = os.path.join(FOLDER, filename)
    with open(path, encoding="utf-8") as f:
        yield from csv.DictReader(f)


def main():
    words = {}
    for filename in SOURCES:
        for row in load_rows(filename):
            headword = (row.get("headword") or "").strip()
            level = (row.get("CEFR") or "").strip().upper()
            if not headword or level not in LEVEL_RANK:
                continue
            # Multi-word phrases ("air conditioning") never match this
            # app's single-token tokenizer, so they'd just be dead data.
            if " " in headword:
                continue
            pos = (row.get("pos") or "").strip().lower()
            pos = POS_FIXES.get(pos, pos)
            # A few headwords list spelling variants together, e.g.
            # "acknowledgment/acknowledgement" or "a.m./A.M./am/AM" -
            # each variant is a real, independently typed word form.
            for variant in headword.split("/"):
                word = variant.strip().lower()
                if not word:
                    continue
                by_pos = words.setdefault(word, {})
                existing = by_pos.get(pos)
                if existing is None or LEVEL_RANK[level] < LEVEL_RANK[existing]:
                    by_pos[pos] = level

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=0, sort_keys=True)

    print(f"Wrote {len(words)} words to {OUT_PATH}")


if __name__ == "__main__":
    main()
