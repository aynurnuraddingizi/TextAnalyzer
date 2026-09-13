# CEFR vocabulary data sources

This folder holds the raw source data `build_cefr_json.py` combines into
`cefr_en.json`, and `build_cefr_es_json.py` collapses into `cefr_es.json`
(both project root) - the files `text_analyzer.py` actually loads at
runtime for CEFR level lookups (A1-C2 for English, A1-C1 for Spanish -
see below). Nothing in this folder ships with the built app; only the
two derived JSON files do, same split as the Russian dictionary's build
script (see the comment above `OFFLINE_DICT_LANGS` in
`text_analyzer.py`).

## English sources

Both files are unmodified copies from
https://github.com/openlanguageprofiles/olp-en-cefrj (retrieved
2026-09-12), which in turn credits http://www.cefr-j.org/ and
http://www.octanove.com/.

- **cefrj-vocabulary-profile-1.5.csv** - CEFR-J Vocabulary Profile,
  version 1.5, A1-B2. Compiled by Yukio Tono, Tokyo University of
  Foreign Studies.
- **octanove-vocabulary-profile-c1c2-1.0.csv** - Octanove Vocabulary
  Profile, version 1.0, C1-C2. Created by Octanove Labs.

## English licenses

**CEFR-J Vocabulary Profile (A1-B2):** "can be used for research and
commercial purposes with no charge, provided that you cite the dataset
properly." Copyright Tono Laboratory, Tokyo University of Foreign
Studies. Cite as:

> The CEFR-J Wordlist Version 1.5. Compiled by Yukio Tono, Tokyo
> University of Foreign Studies. Retrieved from
> http://www.cefr-j.org/download.html

**Octanove Vocabulary Profile (C1-C2):** licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Commercial use is permitted; the license is share-alike, so any
distributed derivative that includes this data - including
`cefr_en.json`, which merges rows from both files - must itself be
made available under CC BY-SA 4.0 and credit Octanove Labs. This
applies only to `cefr_en.json` itself (and anything derived from it),
not to the rest of the application.

## Spanish source

**ELELex.tsv** - from the [CEFRLex project](https://cental.uclouvain.be/cefrlex/)
(CENTAL, Université catholique de Louvain), the same research group and
methodology behind EFLLex (English) and FLELex (French) — "a CEFR-graded
lexical resource for Spanish as a foreign language." Retrieved
2026-09-13 from https://cental.uclouvain.be/cefrlex/static/resources/es/ELELex.tsv
(14,290 rows). Unlike the English sources above, ELELex does NOT assign
one level directly — each row instead gives one (word, part-of-speech)
pair's own normalized frequency AND document count at each of five
levels (a1/a2/b1/b2/c1 — **there is no c2 tier in this data at all**,
a real gap in the source itself, not something lost in processing).
Collapsing that per-level distribution into the single level
`cefr_es.json` actually stores is real interpretive work, done in
`build_cefr_es_json.py` and fully justified in that script's own
docstring — read it before changing the method. In short: a word's
level is the EARLIEST level at which it appears in at least 2 separate
documents (not just frequency > 0, which a single stray mention can
produce) — confirmed directly against real words ("casa"/house,
"tribunal", "biodiversidad", "sin embargo" all land where a fluent
speaker would expect) before this specific threshold was settled on.

## Spanish license

**ELELex** is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
— Attribution, **NonCommercial**, ShareAlike. This is a real, meaningful
difference from every other license already in this project: it is the
ONLY bundled resource that restricts commercial use at all. Concretely:

- `cefr_es.json` (and anything derived from it) may be used and
  redistributed for free/non-commercial purposes, exactly what this
  app currently is, but **may not be sold or otherwise used
  commercially** without separate permission from CENTAL/UCLouvain.
- Share-alike applies too: a distributed derivative including this data
  must itself be CC BY-NC-SA 4.0 and credit the CEFRLex project.
- If this application's distribution model ever changes to something
  commercial, `cefr_es.json` specifically (not the rest of the app)
  would need to be dropped or replaced first.

Citation (per the CEFRLex project's own request; ELELex's own dedicated
reference article was still forthcoming as of the retrieval date above):

> CEFRLex project, CENTAL, Université catholique de Louvain.
> https://cental.uclouvain.be/cefrlex/elelex/

See `THIRD_PARTY_LICENSES.txt` at the project root for how this is
recorded there.
