# Phrase/idiom dictionary source

This folder holds the source data and build script `build_phrase_dictionary.py`
uses to produce `phrase_en.json`/`phrase_es.json` (project root) — the
files `text_analyzer.py`'s `phrase_dictionary_lookup()` actually loads at
runtime to recognize real idioms, phrasal verbs, and fixed expressions
in a loaded book. Nothing in this folder ships with the built app; only
the derived `phrase_<lang>.json` files do — same split as `cefr_data/`
and the German/Arabic/Turkish offline dictionaries' own build scripts
(see the comment above `OFFLINE_DICT_LANGS` in `text_analyzer.py`).

## Source

The raw data is kaikki.org's structured, machine-readable extraction of
Wiktionary content ("Wiktextract", by Tatu Ylonen) — the exact same
source already used for this project's `dict_ar.sqlite3`/`dict_tr.sqlite3`
(see `THIRD_PARTY_LICENSES.txt`), just filtered for a different
`lang_code` and for multi-word headwords specifically rather than every
word.

Retrieved 2026-09-13 from the genuine multi-language raw dump:
`https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz`
(~2.7GB compressed, ~1.49 million entries covering every word Wiktionary
documents in every language, each with English-language glosses — the
build script filters this down per-language). This raw file is not kept
here (far too large); re-download it from the URL above to rerun the
build script or add another language.

**Note, so this mistake isn't repeated**: kaikki.org's per-language
"dictionary" pages (e.g. `kaikki.org/dictionary/English/kaikki.org-
dictionary-English.jsonl`, this project's ORIGINAL source for
`phrase_en.json`) are each already pre-filtered to just that one
language's own entries — confirmed directly when building `phrase_es.json`
against the "English" one first came back with zero results. That never
affected `phrase_en.json` (it only ever wanted `lang_code == "en"`
anyway, which that file is 100% of), but building any OTHER language's
phrase dictionary needs the real multi-language URL above.

## What the build script keeps

- One language's entries only (`lang_code == "en"` for phrase_en.json,
  `"es"` for phrase_es.json).
- Multi-word headwords only (the word contains a space) — genuine
  editorially-catalogued phrases, not single words.
- Excludes pure inflected-form cross-references (e.g. "gave up"
  redirecting to "give up") — only the base form's own real entry is
  kept; `text_analyzer.py`'s lookup applies the same lemma-fallback
  matching already used for CEFR levels so an inflected form in a book
  ("gave up", "looking after") still finds its dictionary's base entry.
- Excludes Wiktionary's own "sum-of-parts"/"translation hub" entries —
  a documented Wiktionary convention for common word pairs its own
  editors explicitly do NOT consider idiomatic (their gloss says so
  outright: "Used other than figuratively or idiomatically: see want,
  to."). Confirmed directly this matters: without excluding these,
  common combinations like "want to" or "younger brother" were showing
  up as if Wiktionary had called them idioms, when it explicitly said
  the opposite.

Each kept entry records its part of speech, its first non-disclaimed
sense's definition (Wiktionary's own listed order — same "first listed
sense" choice `text_analyzer.py`'s `define()` already makes for English
WordNet lookups), and whether Wiktionary tagged that sense
`"idiomatic"` — a real editorial judgment, not inferred.

**Known limitation, accepted rather than chased further**: a phrase
with multiple senses uses whichever one is first/idiomatic in
Wiktionary's own listing, which occasionally isn't the sense a specific
book is actually using (e.g. "want to" has a dialectal "about to" sense
that can surface even for an ordinary "want" + "to [verb]" combination)
— the same kind of imprecision this app's WordNet-based single-word
definitions already have and already document, not a new problem
specific to phrases.

**Spanish-specific limitation**: `phrase_dictionary_lookup()` only
applies its inflected-form lemma fallback for English — `word_family_root()`
is built on WordNet's English-only `morphy` analyzer (see its own
docstring in `text_analyzer.py`), with no Spanish equivalent bundled
here, so `phrase_es.json` only matches a book's EXACT spelling of a
headword. Many Spanish idioms are wholly invariant ("sin embargo", "de
vez en cuando") and match regardless; one built around a conjugating
verb ("tomar el pelo") only matches that bare infinitive spelling, not
"le tomó el pelo" or "le están tomando el pelo".

## License

Wiktionary text content is dual-licensed under the Creative Commons
Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0) and the
GNU Free Documentation License (GFDL) version 1.1 or later — identical
terms to `dict_ar.sqlite3`/`dict_tr.sqlite3` already bundled with this
app; see `THIRD_PARTY_LICENSES.txt` for the full notice and attribution
this project already gives that source.

  - CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/legalcode
  - GFDL: https://www.gnu.org/licenses/fdl-1.3.html
  - Wiktionary copyright/licensing info: https://en.wiktionary.org/wiki/Wiktionary:Copyrights

Attribution, per Wiktionary's own copyright page, is satisfied by a
link back to the source: https://en.wiktionary.org and
https://kaikki.org/dictionary/English/
