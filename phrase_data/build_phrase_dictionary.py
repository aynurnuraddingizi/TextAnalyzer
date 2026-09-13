"""One-time build script: filters kaikki.org's raw, multi-language
wiktextract data (Tatu Ylonen's structured extraction of Wiktionary
content, CC BY-SA 4.0 + GFDL — see README.md in this folder) down to
real, editorially-catalogued multi-word entries for ONE language — genuine
idioms, phrasal verbs, and fixed expressions Wiktionary contributors gave
their own dictionary entry to — and writes the compact result to
phrase_<lang>.json at the project root, the file phrase_analyzer.py's
real-phrase matching actually loads at runtime. Same split as
cefr_data/build_cefr_json.py and the German/Arabic/Turkish dictionaries'
own build scripts: source/build material lives here, only the derived
resource ships with the app.

The raw file (~2.7GB compressed / ~23GB uncompressed, one JSON object
per line — every sense of every word in every language Wiktionary
documents, each with English-language glosses) is NOT kept in this
folder or bundled with the app; it's downloaded fresh by re-running this
script, from:
    https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz

Confirmed directly (building this script's Spanish output initially
came back with 0 entries, which is what caught this): kaikki.org's
per-language "dictionary" pages, e.g. kaikki.org/dictionary/English/
kaikki.org-dictionary-English.jsonl — this project's ORIGINAL source for
phrase_en.json, and still a perfectly valid one FOR ENGLISH — are each
already filtered down to just that one language's own entries (lang_code
"en" only, in that file's case); they are not the multi-language raw
dump this docstring used to claim they were. That distinction never
affected phrase_en.json (it only ever wanted lang_code=="en" anyway,
which that file is 100% of), but it means building any OTHER language's
phrase_<lang>.json needs THIS url instead — the genuine multi-language
raw-wiktextract-data.jsonl.gz above, not a same-shaped-sounding
per-language "dictionary" URL.

Run it again only if the source changes, or to add another language.
Accepts either the plain .jsonl or kaikki.org's .gz directly (streamed,
never fully decompressed to disk):
    python phrase_data/build_phrase_dictionary.py /path/to/raw-wiktextract-data.jsonl.gz en
    python phrase_data/build_phrase_dictionary.py /path/to/raw-wiktextract-data.jsonl.gz es

Filtering logic:
  - lang_code == the requested code (e.g. "en", "es") — the raw file
    covers words FROM every language, each with English-language
    definitions; this app's other bundled Wiktionary-derived
    dictionaries, dict_ar.sqlite3/dict_tr.sqlite3, instead filter for
    Arabic/Turkish lang_code — same source, different filter.
  - word contains a space (a real multi-word entry, not a single word).
  - NOT purely an inflected form of another entry: a word like "gave
    up" exists in the raw data only as a "form-of" cross-reference to
    the base entry "give up" (see below for why lookups still need to
    reach these — the point here is which entry OWNS the real
    definition, not which spellings redirect to it).

Multiple raw entries can share the same headword+pos (e.g. "give up"
has separate verb-vs-noun homograph entries) — every one is kept, each
under its own "pos" key, mirroring cefr_en.json's own {word: {pos:
level}} shape (see build_cefr_json.py) rather than collapsing them.

Each entry's definition is its first sense's first gloss (Wiktionary's
own listed order — same "first listed sense" choice text_analyzer.py's
define() makes for English WordNet lookups, for the same reason: it's
the sense Wiktionary's own editors put first, not a guess). An
"idiomatic" flag records whether that sense carries Wiktionary's own
"idiomatic" tag (a real editorial judgment, not inferred) — shown in
the app to distinguish "editors explicitly called this a figurative
idiom" from "this is merely its own real dictionary entry" (a genuine,
useful distinction: "give up" and "look after" are both entirely real,
common phrasal verbs, just not literally tagged "idiomatic" the way
"kick the bucket" is).
"""
import gzip
import json
import sys
import os

FOLDER = os.path.dirname(os.path.abspath(__file__))


def _open_raw(path):
    # Accepts either the plain .jsonl or the .gz kaikki.org also offers
    # (compressed ~2.7GB vs. ~23GB uncompressed) — streaming straight
    # through the compressed file line-by-line avoids ever needing the
    # uncompressed copy on disk at all, not just saving the download.
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def main(raw_path, lang_code):
    out_path = os.path.join(os.path.dirname(FOLDER), f"phrase_{lang_code}.json")
    phrases = {}
    total = 0
    kept = 0
    with _open_raw(raw_path) as f:
        for line in f:
            total += 1
            if total % 250000 == 0:
                print(f"  ...{total} lines processed, {kept} phrase entries kept so far")
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("lang_code") != lang_code:
                continue
            word = (entry.get("word") or "").strip().lower()
            if " " not in word:
                continue
            pos = (entry.get("pos") or "").strip()
            if not pos:
                continue
            senses = entry.get("senses") or []
            # Skip an entry that is ENTIRELY a form-of cross-reference
            # (e.g. "gave up" pointing back at "give up") — every sense
            # would otherwise need its own redirect-following logic, and
            # the base form's own entry already carries the real
            # definition this app actually wants to show.
            real_senses = [
                s for s in senses
                if s.get("glosses") and "form-of" not in (s.get("tags") or [])
            ]
            if not real_senses:
                continue
            # Wiktionary deliberately creates "sum-of-parts"/"translation
            # hub" entries for very common word pairs it explicitly does
            # NOT consider idiomatic — their own gloss says so outright
            # ("Used other than figuratively or idiomatically: see want,
            # to."). Confirmed directly (English): ~3,000 multi-word
            # entries carry exactly this self-disclaiming gloss, and
            # because they're generated for common combinations, they
            # show up disproportionately often in real book text —
            # "want to", "younger brother" — actively misleading a
            # reader into thinking Wiktionary called these idioms when
            # it explicitly said the opposite. Skipped here rather than
            # filtered later, preferring any OTHER real sense the same
            # entry might have.
            non_sop_senses = [
                s for s in real_senses
                if "other than figuratively or idiomatically" not in s["glosses"][0]
            ]
            if not non_sop_senses:
                continue
            first = non_sop_senses[0]
            gloss = first["glosses"][0].strip()
            if not gloss:
                continue
            idiomatic = any("idiomatic" in (s.get("tags") or []) for s in real_senses)
            by_pos = phrases.setdefault(word, {})
            # A homograph rarely repeats the exact same (word, pos) pair
            # in this raw data; on the rare occasion it does, the first
            # one encountered (Wiktionary's own file order) wins, same
            # "don't overwrite with a later duplicate" choice as
            # build_cefr_json.py's level-conflict handling, just without
            # a level to compare — there's no principled way to prefer
            # one arbitrary duplicate's gloss over another's.
            if pos not in by_pos:
                by_pos[pos] = {"definition": gloss, "idiomatic": idiomatic}
                kept += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=0, sort_keys=True)
    print(f"Processed {total} raw lines.")
    print(f"Wrote {len(phrases)} unique multi-word headwords ({kept} word+pos entries) to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_phrase_dictionary.py /path/to/kaikki-english.jsonl <lang_code>")
        print("  e.g.: python build_phrase_dictionary.py raw.jsonl en")
        print("        python build_phrase_dictionary.py raw.jsonl es")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
