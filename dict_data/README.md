# Spanish offline dictionary source

This folder holds the source/build script for `dict_es.sqlite3`
(project root) — the bundled offline Spanish->English dictionary
`text_analyzer.py`'s `build_glossary()` falls back to for a word NLTK's
Spanish WordNet doesn't have a match for. Nothing in this folder ships
with the built app; only the derived `dict_es.sqlite3` does — same
split as `cefr_data/` and `phrase_data/`.

## Why this exists

Spanish already had real dictionary support in this app via NLTK's
bundled Spanish WordNet (part of the Open Multilingual Wordnet /
Multilingual Central Repository, see `WORDNET_LANG_CODES` in
`text_analyzer.py`). Confirmed directly from a real book run, though:
that coverage is genuinely sparse — 3,967 of 5,995 unique words (66%)
came back with no definition at all. This dictionary fills exactly that
gap. It's consulted as a FALLBACK, not a replacement: `build_glossary()`
tries WordNet first for every word, and only falls through to this
dictionary for the words WordNet had nothing for — unlike Arabic (see
`OFFLINE_DICT_PRIORITY_LANGS` in `text_analyzer.py`), Spanish's WordNet
answers, where they exist at all, are reliable, so there's no reason to
second-guess them.

## Source

The raw data is kaikki.org's structured, machine-readable extraction of
Wiktionary content ("Wiktextract", by Tatu Ylonen) — the exact same
source, and the exact same raw file, already used for this project's
`phrase_es.json` (see `phrase_data/README.md`) — just filtered for
single words instead of multi-word phrases.

Retrieved 2026-09-13 from the genuine multi-language raw dump:
`https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz`
(~2.7GB compressed — see `phrase_data/README.md`'s own note on why this
specific URL, and not a same-shaped-sounding per-language "dictionary"
URL, is the one that actually contains every language's entries). This
raw file is not kept here; re-download it from the URL above to rerun
the build script. The result: 748,963 unique Spanish headwords — a
large number mostly explained by Spanish's rich verb conjugation
(Wiktionary documents a separate "form-of" entry for nearly every
conjugated form of nearly every verb it covers, and this dictionary
deliberately keeps those — see below).

## What the build script keeps

- Spanish-language entries only (`lang_code == "es"`).
- Single-word headwords only (the word contains no space) — multi-word
  entries are `phrase_es.json`'s job, kept separate.
- Every real sense's gloss, not just the first (unlike the phrase
  dictionary) — a plain word-lookup dictionary benefits from showing a
  polysemous word's full breadth of meaning; multiple raw entries for
  the same headword under different parts of speech are merged into
  this same list, since `dict_es.sqlite3`'s schema (`word -> definition`,
  matching `dict_ar.sqlite3`/`dict_tr.sqlite3`/`dict_ru.sqlite3` exactly)
  has no per-part-of-speech structure to keep them apart in.
- Deliberately KEEPS "form-of" entries (e.g. "gatos" -> "plural of
  gato") that the phrase dictionary deliberately excludes: a phrase's
  base-form entry already has the real definition, reachable via
  lemma-fallback matching, so a redirect entry was pure noise there. A
  single Spanish word has no equivalent fallback available
  (`word_family_root()` is English-morphy-only, see its own docstring)
  — without keeping "gatos" -> "plural of gato" itself, that word would
  simply be undefined, and the form-of gloss is genuinely useful
  information on its own.
- Same Wiktionary self-disclaiming-gloss filter as the phrase
  dictionary, applied per-sense.
- A single sense's gloss is sometimes split across more than one string
  in the raw data (an inflection template quirk specific to this
  language's Wiktionary content, e.g. `["inflection of substantivar:",
  "first/third-person singular present subjunctive"]` for ONE sense,
  not two) — every piece is joined, not just the first, which alone
  would be a dangling, unreadable fragment.

Every kept word's glosses are deduplicated (exact string match) and
joined with `"; "`, matching `dict_tr.sqlite3`'s own existing
formatting convention.

**Known limitation, accepted rather than chased further**: the raw
data includes some genuinely archaic/obsolete Spanish (e.g. "far" and
"fablar" as historical spellings of "hacer"/"hablar") alongside modern
vocabulary, since Wiktionary documents a language's full history, not
just its current form — a rare word from this dictionary occasionally
being an archaism rather than modern usage is accepted the same way
this app already accepts WordNet's own occasional obscure-sense-first
ordering elsewhere.

## License

Wiktionary text content is dual-licensed under the Creative Commons
Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0) and the
GNU Free Documentation License (GFDL) version 1.1 or later — identical
terms to `dict_ar.sqlite3`/`dict_tr.sqlite3`/`phrase_es.json` already
bundled with this app; see `THIRD_PARTY_LICENSES.txt` for the full
notice and attribution this project already gives that source.

  - CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/legalcode
  - GFDL: https://www.gnu.org/licenses/fdl-1.3.html
  - Wiktionary copyright/licensing info: https://en.wiktionary.org/wiki/Wiktionary:Copyrights

Attribution, per Wiktionary's own copyright page, is satisfied by a
link back to the source: https://en.wiktionary.org and
https://kaikki.org/dictionary/
