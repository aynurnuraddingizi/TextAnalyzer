"""One-time build script: filters kaikki.org's raw, multi-language
wiktextract data (same source as phrase_data/build_phrase_dictionary.py
— see that file's own docstring for the exact URL and the "which
kaikki.org URL is actually the multi-language one" caveat it documents)
down to real, single-word Spanish headwords with real English-language
definitions, and writes them to dict_es.sqlite3 at the project root —
the same schema, and the same "real Wiktionary dictionary, not a guess"
sourcing, as this project's existing dict_ar.sqlite3/dict_tr.sqlite3/
dict_ru.sqlite3 (see THIRD_PARTY_LICENSES.txt for those, and the
comment above OFFLINE_DICT_LANGS in text_analyzer.py).

Why Spanish needs this AT ALL, despite already having WordNet coverage
(see WORDNET_LANG_CODES in text_analyzer.py): confirmed directly from
real usage — a real book's vocabulary run came back with 3,967 of 5,995
unique words (66%) undefined, because NLTK's bundled Spanish WordNet
(part of the Multilingual Central Repository, distributed via omw-1.4)
is real but genuinely sparse — nowhere near as complete as English
WordNet. This dictionary is used as a FALLBACK for exactly those words
(see build_glossary()'s `offline_dict_lang` parameter) — WordNet is
still tried first (Spanish is NOT added to OFFLINE_DICT_PRIORITY_LANGS
the way Arabic is, since Spanish's WordNet coverage, unlike Arabic's,
is reliable when present — the problem is purely its gaps, not its
quality), so this only ever fills in words WordNet had nothing for.

Run it again only if the source changes:
    python dict_data/build_dict_es.py /path/to/raw-wiktextract-data.jsonl.gz

Filtering logic (same spirit as build_phrase_dictionary.py's, mirrored
for single words instead of phrases):
  - lang_code == "es".
  - word does NOT contain a space (multi-word entries are
    phrase_es.json's job, not this dictionary's — kept separate so a
    real idiom's own curated definition isn't diluted by also being
    dumped, sense-by-sense, into a plain word lookup).
  - Every real sense's gloss is kept (unlike the phrase dictionary,
    which keeps only the first) — a single-word dictionary entry
    benefits from showing the breadth of a polysemous word's meanings
    (e.g. "banco": bench; bank; school of fish), where a phrase entry's
    whole point was picking the ONE real idiomatic sense. Multiple raw
    entries for the same headword (different parts of speech) are
    merged into this same list rather than kept separate — this
    dictionary's schema (word -> definition, matching dict_ar.sqlite3/
    dict_tr.sqlite3/dict_ru.sqlite3 exactly) has no per-POS structure.
  - Unlike the phrase dictionary, "form-of" entries (e.g. "gatos" ->
    "plural of gato") are KEPT, not excluded: for a phrase, a form-of
    entry was redundant noise (the base form's own entry already had
    the real definition, and phrase lookups already apply lemma-
    fallback matching). A single Spanish word has NO equivalent
    fallback available (word_family_root() is English-only, see its
    own docstring) — "gatos" with no direct entry would otherwise be
    entirely undefined, and "plural of gato" is genuinely useful
    information a reader can act on immediately, not a case of the app
    quietly hiding something better it already has.
  - Same Wiktionary "sum-of-parts"/self-disclaiming-gloss filter as the
    phrase dictionary, for the same reason, applied per-sense rather
    than per-entry (a word can have one real sense and one
    disclaimed one; only the disclaimed one is dropped).

Glosses for one headword are deduplicated (case-sensitive exact match
only — cheap, catches the common case of the same gloss text appearing
under two different part-of-speech entries) and joined with "; ",
matching dict_tr.sqlite3's own existing join convention exactly.
"""
import gzip
import json
import os
import sqlite3
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(os.path.dirname(FOLDER), "dict_es.sqlite3")


def _open_raw(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def main(raw_path):
    words = {}  # word -> [gloss, gloss, ...] in first-seen order
    total = 0
    with _open_raw(raw_path) as f:
        for line in f:
            total += 1
            if total % 500000 == 0:
                print(f"  ...{total} lines processed, {len(words)} Spanish headwords kept so far")
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("lang_code") != "es":
                continue
            word = (entry.get("word") or "").strip().lower()
            if not word or " " in word:
                continue
            senses = entry.get("senses") or []
            glosses = words.setdefault(word, [])
            for sense in senses:
                sense_glosses = sense.get("glosses")
                if not sense_glosses:
                    continue
                # Confirmed directly: unlike English/Turkish's own data,
                # a single Spanish SENSE's "glosses" list is sometimes
                # split across multiple strings for one inflection
                # template — e.g. ["inflection of substantivar:",
                # "first/third-person singular present subjunctive"] is
                # ONE sense's gloss, not two — so every element is
                # joined, not just the first (which alone would be a
                # dangling, unreadable "inflection of substantivar:").
                gloss = " ".join(g.strip() for g in sense_glosses if g.strip())
                if not gloss or "other than figuratively or idiomatically" in gloss:
                    continue
                if gloss not in glosses:
                    glosses.append(gloss)

    print(f"Processed {total} raw lines.")
    print(f"Writing {len(words)} unique Spanish headwords to {OUT_PATH}")

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    conn = sqlite3.connect(OUT_PATH)
    conn.execute("CREATE TABLE dict (word TEXT PRIMARY KEY, definition TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO dict (word, definition) VALUES (?, ?)",
        ((word, "; ".join(glosses)) for word, glosses in words.items() if glosses),
    )
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_dict_es.py /path/to/raw-wiktextract-data.jsonl.gz")
        sys.exit(1)
    main(sys.argv[1])
