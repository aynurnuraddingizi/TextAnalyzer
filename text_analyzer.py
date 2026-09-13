"""
Vocabulary Glossary Builder — extracts every unique word from a PDF or
EPUB, in any language, and looks up a dictionary definition for each one.

WHAT IT DOES:
  1. Reads any .pdf or .epub file you point it at (a book, article, etc.)
  2. Extracts every distinct word in it (any script/language, deduplicated)
  3. Detects the book's language once from the full text
  4. English words: looked up in NLTK's WordNet dictionary (offline,
     no network use). Non-English words: WordNet where it has native
     coverage, plus bundled offline dictionaries for German/Arabic/
     Turkish (see OFFLINE_DICT_LANGS). Words nothing matches are left
     undefined — there's no online translate-to-English fallback (tried
     and removed: every free backend available fell short of this app's
     own bar for dictionary quality, see build_glossary's docstring).
  5. Saves a glossary file, sorted alphabetically, plus a list of any
     words nothing could be found for (names, jargon, typos, etc.)

HOW TO RUN:
    python text_analyzer.py
  (it will ask you for the path to a .pdf or .epub file)

REQUIRES:
    pip install pypdf nltk ebooklib beautifulsoup4 langdetect gtts
  The first run also downloads NLTK's WordNet data automatically
  (needs an internet connection once; cached locally after that).

  Cross-platform: everything above works unchanged on Windows, macOS,
  and Linux. The one platform-specific piece is speak_word()'s audio
  playback (used for the pronunciation feature) — Windows and macOS
  play back with tools every install already has, but Linux has no
  single standard command-line MP3 player, so it needs one of mpg123,
  ffplay (part of ffmpeg), mpv, or vlc installed, e.g.
  `sudo apt install mpg123`. Everything else in this file needs nothing
  platform-specific.
"""

import ctypes
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

from pypdf import PdfReader
from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import cmudict, stopwords
from langdetect import detect, LangDetectException
from gtts import gTTS

LANGUAGE_NAMES = {
    "en": "English", "tr": "Turkish", "az": "Azerbaijani", "de": "German",
    "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "nl": "Dutch", "sv": "Swedish", "pl": "Polish",
    "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean",
}

# ISO 639-1 (as used by langdetect throughout this app) -> ISO 639-3
# WordNet language code, for languages NLTK's bundled Open Multilingual
# Wordnet data actually has native coverage for. Confirmed by direct
# enumeration (wn.langs(), after downloading omw-1.4) that this covers
# far more than "spa" and "arb" alone — closer to 30 languages, including
# "fra"/French — but this app only lists a language here once its
# define() output has actually been spot-checked against real vocabulary
# (see define()'s docstring for how French's sense-picking was verified
# and, in one case, tuned). German is deliberately absent — there is no
# free/redistributable German WordNet (GermaNet is licensed separately);
# it gets its own offline dictionary instead, see OFFLINE_DICT_LANGS
# below. Russian is absent because omw-1.4 simply has no Russian data at
# all (confirmed: wn.synsets(..., lang="rus") raises "Language is not
# supported", not just "no results") — it would need an offline
# dictionary of its own, same as German, to ever be added here.
WORDNET_LANG_CODES = {"en": "eng", "es": "spa", "ar": "arb", "fr": "fra"}

# ISO 639-1 -> bundled SQLite offline-dictionary filename, used as a
# fallback for words WordNet doesn't have (Arabic, Spanish), or as the
# sole offline source for languages with no WordNet data at all (German,
# Turkish, Russian). "de" is built from FreeDict's deu-eng (GPL-2,
# 382,791 words). "ar", "tr", "ru", and "es" are NOT built from
# FreeDict: FreeDict's only free Arabic dictionary was tested and found
# to have serious quality problems on common words (e.g. "book" came
# back as "Casebook", "water" as "Miaowed"), and its Turkish dictionary
# was real but tiny (1,023 words) — both were rejected as sources. "ar",
# "tr", "ru", and "es" are instead built from kaikki.org's structured
# extraction of Wiktionary's entries for each language (Wiktextract;
# text content is Wiktionary, dual-licensed CC BY-SA 4.0 / GFDL — see
# THIRD_PARTY_LICENSES.txt): "ar" is 26,572 words, "tr" is 41,264 words,
# "ru" is 426,835 words, "es" is 748,963 words (see dict_data/README.md
# for why that count is so much larger — mostly Spanish's own rich verb
# conjugation), all spot-checked against common vocabulary before
# shipping (build scripts for "ar"/"tr"/"ru" live in the project's
# history, not bundled with the app itself; "es" — added later, once
# this project started keeping build scripts alongside their output —
# has its kept at dict_data/build_dict_es.py). See OFFLINE_DICT_LANGS
# below and WORDNET_LANG_CODES above for how Arabic/Spanish each
# combine both an offline dictionary and WordNet; German/Turkish/
# Russian have no WordNet data at all (see WORDNET_LANG_CODES) so this
# is their only source.
OFFLINE_DICT_LANGS = {
    "de": "dict_de.sqlite3",
    "tr": "dict_tr.sqlite3",
    "ar": "dict_ar.sqlite3",
    "ru": "dict_ru.sqlite3",
    "es": "dict_es.sqlite3",
}

# Languages where the bundled offline dictionary should be tried BEFORE
# WordNet, not after. Verified by direct side-by-side comparison across
# the 60 most common words of a real Quran text: NLTK's Arabic WordNet
# (arb) returned nothing at all for ~75% of them, and for several of the
# rest returned an actively wrong sense that pre-empted a correct one
# already sitting in dict_ar.sqlite3 — e.g. "على" (on/over, a very common
# preposition) came back from WordNet as "raise from a lower to a higher
# position", and "أن" (that, a conjunction) as "place (seeds) in or on
# the ground for future growth". Since build_glossary only consults the
# offline dictionary when WordNet found NOTHING, those WordNet mistakes
# were silently winning. Spanish is deliberately NOT in this set despite
# also having both sources: unlike Arabic's WordNet, Spanish's is real
# and reliable where it has an entry at all — confirmed directly, its
# problem is coverage (a real book came back with 66% of its unique
# words undefined), not wrong answers — so it's used first, with
# dict_es.sqlite3 filling in exactly the gap WordNet leaves rather than
# competing with it.
OFFLINE_DICT_PRIORITY_LANGS = {"ar"}


def app_dir():
    """Directory the app's editable/persistent files live in.

    When frozen by PyInstaller, on-disk files must sit next to the .exe,
    not inside the bundled onefile archive (sys._MEIPASS), so they
    survive updates and are easy for a user to find/back up. Contrast
    with `_resource_dir()`, used for bundled read-only resources.

    On Android there is no ".exe" and no writable directory next to
    this script (python-for-android's own bundle location isn't
    reliably writable) — `android.storage.app_storage_path()` is
    python-for-android's own blessed API for "a real, private,
    writable directory this app owns", used instead. Only ever
    imports `kivy`/`android` when actually running under Kivy on
    Android — the Windows desktop build never has either installed,
    and this branch must never execute or even attempt the import
    there.
    """
    try:
        from kivy.utils import platform as _kivy_platform
    except ImportError:
        _kivy_platform = None
    if _kivy_platform == "android":
        from android.storage import app_storage_path
        return app_storage_path()
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _resource_dir():
    """Directory bundled read-only resources (the offline dictionaries)
    live in. When frozen by PyInstaller, these are unpacked to a
    temporary directory (sys._MEIPASS) as part of the onefile archive —
    unlike `app_dir()`'s editable files, they're never meant to be edited
    or found next to the .exe, so a different resolution is needed here.
    """
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def offline_dict_available(lang):
    """Whether a bundled offline dictionary exists for this ISO 639-1
    language code (regardless of whether the data file can actually be
    found on disk right now — see `_offline_dict_path`).
    """
    return lang in OFFLINE_DICT_LANGS


def _offline_dict_path(lang):
    filename = OFFLINE_DICT_LANGS.get(lang)
    if not filename:
        return None
    path = os.path.join(_resource_dir(), filename)
    return path if os.path.exists(path) else None


def progress_path():
    return os.path.join(app_dir(), "vocabulary_progress.json")


def empty_language_progress():
    """A fresh, empty progress bucket for one language — the shape every
    entry in load_progress()["languages"] has. Exposed for callers (the
    app, when switching to a book language it's never seen progress for
    before) that need to create one on the fly, not just this module.
    """
    return {"known": set(), "learning": set(), "schedule": {}, "learned_at": {}}


def load_progress():
    """Load the reader's long-term vocabulary progress, kept SEPARATELY
    per book language (ISO 639-1 code, e.g. "en"/"de") — every word
    they've marked "known" or "still learning" in Study mode, across
    every book they've ever opened in that language, not just the
    current session, plus each word's spaced-repetition schedule (see
    review_word()) and the date each was first ever marked known (see
    mark_known()). This is what lets a new book's glossary recognize
    words the reader has already mastered instead of treating every book
    as a blank slate — scoped to the SAME language, so English progress
    and German progress (say) never mix into one undifferentiated pile
    that would misleadingly inflate either one's "reading readiness".

    Returns {"languages": {lang: {"known": set(), "learning": set(),
    "schedule": dict, "learned_at": dict}}, "reset_backups": {lang:
    {...}}, "grammar": {"known": set(), "learning": set(), "schedule":
    dict, "learned_at": dict}} — see backup_before_reset()/
    restorable_backup() for what "reset_backups" holds. A missing or
    corrupt file quietly starts fresh rather than erroring; a missing
    "schedule"/"learned_at" for a language also covers a progress file
    saved before spaced repetition or progress-tracking dates existed.

    "grammar" is a SINGLE bucket, not per-language like "languages" —
    grammar structure detection is English-only (see
    grammar_analyzer.analyze_grammar()'s docstring), so there's never a
    second language's grammar progress to keep separate from a first
    one; same shape as one language's own bucket purely because it's the
    same know/learning/schedule/learned_at concept applied to rule ids
    instead of words, not because grammar is secretly per-language too.
    A file from before this key existed simply has none, and
    `data.get("grammar", {})` below reads that the same way a
    pre-per-language file's missing "schedule" already does.

    A file saved by a version of this app before per-language tracking
    existed (a flat "known"/"learning" with no "languages" key) is
    migrated once, treated entirely as English — every word marked known
    in this app's history up to that point genuinely was from an English
    book, confirmed directly from this project's own session history, so
    this isn't a guess so much as a reflection of the only real data that
    ever existed before this file format did.
    """
    path = progress_path()
    if not os.path.exists(path):
        return {"languages": {}, "reset_backups": {}, "grammar": empty_language_progress()}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"languages": {}, "reset_backups": {}, "grammar": empty_language_progress()}

    def _read_bucket(entry):
        return {
            "known": set(entry.get("known", [])),
            "learning": set(entry.get("learning", [])),
            "schedule": entry.get("schedule", {}),
            "learned_at": entry.get("learned_at", {}),
        }

    if "languages" not in data:
        languages = {}
        if data.get("known") or data.get("learning") or data.get("schedule") or data.get("learned_at"):
            languages["en"] = _read_bucket(data)
        return {"languages": languages, "reset_backups": {}, "grammar": empty_language_progress()}

    languages = {lang: _read_bucket(entry) for lang, entry in data.get("languages", {}).items()}
    reset_backups = prune_expired_backups(data.get("reset_backups", {}))
    grammar = _read_bucket(data["grammar"]) if "grammar" in data else empty_language_progress()
    return {"languages": languages, "reset_backups": reset_backups, "grammar": grammar}


def save_progress(languages, reset_backups=None, grammar=None):
    """Persist every language's known/still-learning word sets, spaced-
    repetition schedule, and learned-on dates, plus any pending reset
    backups and the single grammar-progress bucket, to disk.

    `languages` is {lang: {"known": set, "learning": set, "schedule":
    dict, "learned_at": dict}} — the whole load_progress()["languages"]
    structure, not just the currently active language, since the file
    holds every language's progress at once. `reset_backups`, if given,
    is load_progress()["reset_backups"] — omitting it (rather than
    silently defaulting to {}) would erase a pending "undo my last
    reset" backup on any save that doesn't happen to originate from
    reset_progress() itself, so every call site should pass what it
    loaded or was handed, even when unchanged. `grammar`, if given, is
    load_progress()["grammar"] — same reasoning: omitting it on a save
    that isn't about grammar at all would silently erase grammar
    progress, so every call site passes it through unchanged too.
    """
    with open(progress_path(), "w", encoding="utf-8") as f:
        json.dump(
            {
                "languages": {
                    lang: {
                        "known": sorted(entry["known"]), "learning": sorted(entry["learning"]),
                        "schedule": entry.get("schedule") or {}, "learned_at": entry.get("learned_at") or {},
                    }
                    for lang, entry in languages.items()
                },
                "reset_backups": reset_backups or {},
                "grammar": {
                    "known": sorted((grammar or {}).get("known", [])),
                    "learning": sorted((grammar or {}).get("learning", [])),
                    "schedule": (grammar or {}).get("schedule") or {},
                    "learned_at": (grammar or {}).get("learned_at") or {},
                },
            },
            f, ensure_ascii=False, indent=2,
        )


# How long a reset's backup stays restorable (RESTORE PROGRESS in the
# sidebar) before it's treated as expired and pruned on next load.
RESET_BACKUP_DAYS = 7


def backup_before_reset(reset_backups, lang, known, learning, schedule, learned_at):
    """Snapshot one language's progress into `reset_backups` right
    before reset_progress() clears it, so RESTORE PROGRESS can bring it
    back within RESET_BACKUP_DAYS — see restorable_backup(). Overwrites
    any existing backup for `lang`: only the most recent reset needs to
    be undoable, not a deeper history, so an unresolved backup from a
    previous reset simply gets superseded rather than stacking up.
    Mutates and returns `reset_backups`, matching this file's other
    mutate-in-place helpers (review_word(), mark_known()).
    """
    from datetime import datetime, timezone

    reset_backups[lang] = {
        "backed_up_at": datetime.now(timezone.utc).isoformat(),
        "known": sorted(known), "learning": sorted(learning),
        "schedule": schedule, "learned_at": learned_at,
    }
    return reset_backups


def prune_expired_backups(reset_backups):
    """Drop any reset backup older than RESET_BACKUP_DAYS — called from
    load_progress() so an expired "undo" silently stops being offered
    rather than lingering forever, and also safe to call any time the
    caller wants a freshly-verified answer (e.g. right before deciding
    whether to show a RESTORE PROGRESS button, in case the app has
    simply been open, uninterrupted, for longer than the window).
    Mutates and returns `reset_backups`.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=RESET_BACKUP_DAYS)
    for lang in list(reset_backups.keys()):
        try:
            backed_up_at = datetime.fromisoformat(reset_backups[lang]["backed_up_at"])
        except (KeyError, ValueError):
            del reset_backups[lang]
            continue
        if backed_up_at < cutoff:
            del reset_backups[lang]
    return reset_backups


def restorable_backup(reset_backups, lang):
    """The backup for `lang`, if one exists and is still within
    RESET_BACKUP_DAYS of its reset — or None. Callers use this (after a
    fresh prune_expired_backups() call, in case time has passed since
    load_progress() last checked) to decide whether to show/enable a
    RESTORE PROGRESS action.
    """
    prune_expired_backups(reset_backups)
    return reset_backups.get(lang)


def mark_known(learned_at, word):
    """Record the first time `word` was ever marked known, into
    `learned_at` (the dict persisted as progress["learned_at"]) — a
    no-op if it's already recorded, since re-marking an already-known
    word known again isn't new vocabulary growth and shouldn't reset its
    date. Powers "words learned this week/month" (see
    words_learned_since()), which needs to know WHEN a word was learned,
    not just that it eventually was — something known/learning sets
    alone can't answer. Mutates and returns `learned_at`, matching
    review_word()'s style.
    """
    if word not in learned_at:
        from datetime import datetime, timezone

        learned_at[word] = datetime.now(timezone.utc).isoformat()
    return learned_at


def words_learned_since(learned_at, days):
    """How many words have a learned_at date within the last `days`
    days — powers the Progress card's "this week"/"this month" stats.
    Words known from before learned_at existed (or before they were
    marked known through a version of this app that recorded it) simply
    have no entry and are excluded here rather than guessed at — see
    load_progress()'s docstring.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    count = 0
    for timestamp in learned_at.values():
        try:
            when = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        if when >= cutoff:
            count += 1
    return count


# SM-2 (the classic SuperMemo spaced-repetition algorithm, also what Anki's
# default scheduler is based on): each review either grows the gap until the
# next one (a confident recall) or resets it to a short retry (a failed
# one), so Study mode naturally settles into showing well-known words less
# and shaky ones more, instead of a flat random shuffle through everything.
_SM2_DEFAULT_EASE = 2.5
_SM2_MIN_EASE = 1.3


def _sm2_review(entry, quality):
    """Apply one SM-2 review to a schedule entry (None for a
    never-before-seen word), returning the new entry.

    `quality` is 0-5 in the original SM-2 algorithm; this app only ever
    passes 5 ("I know it" — confident recall) or 2 ("still learning" —
    failed) since Study mode's two-button verdict is simpler by design
    than the full 6-point self-grading scale other SRS apps use, and a
    plain word glossary doesn't need that finer distinction to still get
    SM-2's real benefit (failed words come back sooner, solid ones drift
    further apart).
    """
    entry = dict(entry) if entry else {"interval": 0, "ease": _SM2_DEFAULT_EASE, "reps": 0}
    ease = entry["ease"]
    if quality >= 3:
        reps = entry["reps"] + 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = round(entry["interval"] * ease)
    else:
        reps = 0
        interval = 1
    ease = max(_SM2_MIN_EASE, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))

    from datetime import date, timedelta

    due = (date.today() + timedelta(days=interval)).isoformat()
    return {"interval": interval, "ease": round(ease, 2), "reps": reps, "due": due}


def review_word(schedule, word, known):
    """Record one Study-mode verdict on `word` into `schedule` (the dict
    persisted as progress["schedule"]) — known=True is SM-2 quality 5,
    known=False is quality 2 (see _sm2_review). Mutates and returns
    `schedule`, matching add_recent_book()'s style elsewhere in this
    file.
    """
    schedule[word] = _sm2_review(schedule.get(word), 5 if known else 2)
    return schedule


def due_words(schedule, words):
    """Which of `words` are due for review today — past their SM-2
    schedule's due date, or (most commonly, for a first-ever session)
    never scheduled at all. What Study mode now studies first, instead
    of a flat random shuffle through every not-yet-known word.
    """
    from datetime import date

    today = date.today().isoformat()
    return [w for w in words if w not in schedule or schedule[w]["due"] <= today]


def settings_path():
    return os.path.join(app_dir(), "vocabmaster_settings.json")


MAX_RECENT_BOOKS = 8


def load_settings():
    """Load app-level preferences: the light/dark theme choice and the
    list of recently opened books (path/title/opened_at, most-recent
    first). Separate from load_progress()/vocabulary_progress.json,
    which holds learning data, not app preferences. A missing or corrupt
    file quietly starts fresh, same as load_progress().
    """
    path = settings_path()
    if not os.path.exists(path):
        return {"theme": "light", "recent_books": [], "onboarded": False}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "theme": data.get("theme") if data.get("theme") in ("light", "dark") else "light",
            "recent_books": data.get("recent_books", []),
            "onboarded": bool(data.get("onboarded", False)),
        }
    except (OSError, ValueError):
        return {"theme": "light", "recent_books": [], "onboarded": False}


def save_settings(settings):
    with open(settings_path(), "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def add_recent_book(settings, path, title):
    """Return `settings` with `path` moved to the front of recent_books
    (de-duplicated, capped at MAX_RECENT_BOOKS). Does not save to disk —
    call save_settings() with the result when ready.
    """
    from datetime import datetime, timezone

    recent = [b for b in settings.get("recent_books", []) if b.get("path") != path]
    recent.insert(0, {
        "path": path, "title": title,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    })
    settings["recent_books"] = recent[:MAX_RECENT_BOOKS]
    return settings


def remove_recent_book(settings, path):
    """Return `settings` with `path` removed from recent_books, if
    present. Does not save to disk — call save_settings() with the
    result when ready.
    """
    settings["recent_books"] = [
        b for b in settings.get("recent_books", []) if b.get("path") != path
    ]
    return settings


def ensure_wordnet():
    """Make sure NLTK's WordNet dictionary data is downloaded before use.

    English ("wordnet") and the multilingual data ("omw-1.4", which
    Spanish/Arabic lookups need) are checked and downloaded independently
    — confirmed directly that they can be present separately: an install
    that already has English WordNet from before multilingual support
    existed would silently keep missing omw-1.4 forever under the old
    single combined check here, since English lookups alone never touch
    the omw-1.4 resource and so never triggered its download. Attempting
    a non-English lookup without omw-1.4 present raises an unhandled
    LookupError (confirmed directly) rather than failing gracefully, so
    this needs to be caught here, before that ever happens mid-glossary.
    """
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")
    try:
        wn.synsets("test", lang="spa")
    except LookupError:
        nltk.download("omw-1.4")


class NoExtractableTextError(RuntimeError):
    """Raised when a file has (almost) no text a library can pull out of
    it — e.g. a scanned page image, or a PDF where the text was converted
    to vector outline shapes (a real technique some PDF producers use,
    sometimes deliberately for copy-protection). In both cases every page
    still gets visited — nothing is being skipped — there is simply no
    character data in the file for any extraction tool to find, only
    shapes. Distinguishing this from "the app silently missed content"
    matters, so it's raised as its own clear error rather than quietly
    returning an empty result.
    """


def load_pdf_text(filepath):
    """Extract and concatenate the text of every page in a PDF."""
    reader = PdfReader(filepath)
    pages = [page.extract_text() or "" for page in reader.pages]
    total_pages = len(pages)
    empty_pages = sum(1 for p in pages if not p.strip())
    text = "\n".join(pages)
    # A handful of genuinely blank pages (covers, section dividers) is
    # normal. Nearly every page coming back empty means the file has no
    # real text layer at all, not that extraction missed something.
    if total_pages and empty_pages / total_pages > 0.8 and len(text.strip()) < 500:
        raise NoExtractableTextError(
            f"This PDF has no extractable text ({empty_pages}/{total_pages} pages came back "
            "empty). It's likely a scanned image, or its text was converted to vector outline "
            "shapes instead of real characters — both look normal to a viewer but contain no "
            "character data any text-extraction tool (not just this app) can read. "
            "An OCR tool would be needed to recover the words, or try a different copy of the file."
        )
    return text


def load_epub_text(filepath):
    """Extract and concatenate the text of every chapter in an EPUB."""
    book = epub.read_epub(filepath, options={"ignore_ncx": True})
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        chapters.append(soup.get_text(separator="\n"))
    text = "\n".join(chapters)
    if not text.strip():
        raise NoExtractableTextError(
            "This EPUB has no extractable text — its chapters contained no readable text content."
        )
    return text


def load_book_text(filepath):
    """Extract text from a PDF or EPUB, dispatching on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".epub":
        return load_epub_text(filepath)
    return load_pdf_text(filepath)


# Arabic combining diacritics (tanwin, harakat, shadda, sukun), superscript
# alef, Quranic annotation marks, and tatweel. Real Quranic text (Uthmani
# script) carries one of these on almost every letter — confirmed by direct
# testing against a real Quran file, where they're ~85% of all characters.
# Python's \w (used below) does NOT include combining marks, so without
# this a word like "اسْتَوْقَدَ" (istawqada) gets fragmented into pieces at
# every diacritic instead of extracted as one word — silently corrupting
# frequency counts and dictionary lookups for exactly this app's Arabic/
# Qur'an-language use case. They're included in _WORD_PATTERN so a word
# extracts as one run, then stripped by _strip_arabic_diacritics() to form
# the lookup key, so "اسْتَوْقَدَ" normalizes to "استوقد" -- matching how
# plain (undiacritized) Arabic books are written and how dict_ar.sqlite3's
# headwords are stored.
_ARABIC_DIACRITICS_RANGE = "ؐ-ًؚ-ٰٟۖ-ۭـ"
_ARABIC_DIACRITICS_RE = re.compile(f"[{_ARABIC_DIACRITICS_RANGE}]")
_WORD_CHAR = f"(?:[^\\W\\d_]|[{_ARABIC_DIACRITICS_RANGE}])"
_WORD_PATTERN = rf"{_WORD_CHAR}+(?:'{_WORD_CHAR}+)*"
# Extracts runs of Unicode letters (apostrophes allowed for contractions)
# rather than splitting on whitespace or restricting to A-Z. This makes
# it work for any language/script (accented Latin, Cyrillic, Arabic,
# CJK, etc.), not just English. Digits still act as word boundaries,
# since PDF-extracted text can glue page numbers or headers onto
# neighboring words with no space between them (e.g. a running header
# "Systems200Table").


def _strip_arabic_diacritics(word):
    return _ARABIC_DIACRITICS_RE.sub("", word)


def word_frequencies(text):
    """Return {word: occurrence count} for every distinct word in the
    text, lowercased.

    Frequency is the single best guide to which words are worth learning
    first in a foreign-language book: thanks to Zipf's law, a fairly
    small set of the most common words accounts for most of a real
    text's actual word occurrences, so learning those first buys far
    more reading comprehension per word studied than working through the
    list alphabetically or at random.
    """
    counts = {}
    for w in re.findall(_WORD_PATTERN, text, re.UNICODE):
        w = _strip_arabic_diacritics(w.lower())
        if len(w) <= 1:
            continue
        counts[w] = counts.get(w, 0) + 1
    return counts


def extract_unique_words(text):
    """Return every distinct word in the text, lowercased and sorted."""
    return sorted(word_frequencies(text).keys())


def split_sentences(text):
    """Split book text into sentence-like chunks, breaking on ./!/? or a
    "(<number>)" marker, not just Western punctuation — confirmed
    necessary against a real Quran text file, which delimits every verse
    with "(17)"-style ayah numbers and contains no Western sentence
    punctuation at all; without this the entire book is treated as one
    giant "sentence". Shared by build_example_sentences() and
    find_occurrences() so both see identical segmentation, and cacheable
    by a caller (the app keeps one copy per loaded book) since it's the
    same expensive split either way.
    """
    normalized = re.sub(r"\s+", " ", text)
    return re.split(r"(?:(?<=[.!?])|(?<=\d\)))\s+", normalized)


def build_example_sentences(text, words, max_len=350, sentences=None):
    """Find one real example sentence from the book itself for each word
    in `words` — its first occurrence, PREFERRING a reasonably short
    sentence but never simply omitting a word for lack of one.

    Showing a word used the way the book itself actually uses it is far
    more useful to a foreign-language reader than an isolated word plus
    its dictionary translation (word-level translation is frequently
    ambiguous; the surrounding sentence disambiguates it, and seeing real
    usage aids memory) — and unlike an AI-generated example, this needs
    no network call or AI at all, just the book's own text.

    Runs in two passes: the first only considers sentences of at most
    `max_len` characters (likely PDF-extraction artifacts — merged
    paragraphs, tables — get much longer than any real sentence, so
    this keeps the common case a clean, short snippet); a second pass
    then covers every word STILL missing an example with whatever
    sentence it actually occurs in, however long. Earlier, that second
    pass didn't exist — a word whose only occurrence happened to sit in
    a genuinely long (but real) sentence got no example at all, with
    nothing distinguishing "this word doesn't actually occur in a
    short sentence" from "this glossary entry is somehow wrong" to a
    reader looking at a blank example field. Every word that build_
    glossary() found in the book at all is guaranteed a real example
    now, exactly matching where it actually occurs — precision over a
    tidier-looking but silently incomplete result.

    Only the FIRST occurrence of each word is ever surfaced here — for
    every occurrence across the whole book, see find_occurrences()
    (which has the exact same two-tier guarantee, for the same reason).

    `sentences`, if given, is a pre-split list from split_sentences() —
    pass one in if the caller already has it (the app caches one per
    loaded book) to avoid re-splitting the same text twice. Displayed
    sentences keep their original diacritics (not stripped) since that's
    the real, readable text; only the word-matching step below
    normalizes for comparison.
    """
    if sentences is None:
        sentences = split_sentences(text)
    wanted = set(words)
    examples = {}
    for allowed_len in (max_len, None):
        if len(examples) >= len(wanted):
            break
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or (allowed_len is not None and len(sentence) > allowed_len):
                continue
            tokens = {
                _strip_arabic_diacritics(t.lower())
                for t in re.findall(_WORD_PATTERN, sentence, re.UNICODE)
            }
            for word in tokens & wanted:
                if word not in examples:
                    examples[word] = sentence
            if len(examples) >= len(wanted):
                break
    return examples


def _concordance_pattern(query, whole_word):
    """Build a case-insensitive regex matching `query` against raw book
    text, tolerating an Arabic combining diacritic (the same characters
    _strip_arabic_diacritics() removes elsewhere in this file) between,
    and around, every literal character of the query.

    Without this, searching a glossary word's stripped/lowercased
    dictionary key (e.g. "استوقد") could never match the diacritic-marked
    surface form a real Quranic text actually uses (e.g. "اسْتَوْقَدَ")
    — confirmed empirically, a plain substring match finds nothing there
    at all. The extra optional-diacritic gaps are harmless for every
    other script (the diacritics character class simply never matches
    Latin/Cyrillic/CJK/etc. text, so it always matches zero-width there).
    """
    gap = f"[{_ARABIC_DIACRITICS_RANGE}]*"
    pattern = gap.join(re.escape(c) for c in query)
    pattern = f"{gap}{pattern}{gap}"
    if whole_word:
        pattern = rf"\b{pattern}\b"
    return pattern


def mask_word_in_sentence(sentence, word, mask="_____"):
    """Replace every occurrence of `word` in `sentence` with `mask` —
    for Study mode's Writing quiz, which shows the example sentence as a
    clue but obviously can't leave the answer sitting right there in it.

    Reuses _concordance_pattern()'s case-insensitive, diacritic-tolerant,
    whole-word matching (built for find_occurrences()) rather than a
    plain substring replace, for the same reason it exists there: a
    stored dictionary key like "استوقد" needs to still match — and here,
    get correctly masked — its diacritic-marked appearance in the actual
    sentence ("اسْتَوْقَدَ"), not just an exact literal match. If `word`
    doesn't actually appear in `sentence` (shouldn't happen — sentences
    are only ever attached to a word because it was found in them — but
    not worth crashing Study mode over if it somehow did), the sentence
    is returned unchanged rather than raising.
    """
    word = word.strip()
    if not word:
        return sentence
    try:
        regex = re.compile(_concordance_pattern(word, whole_word=True), re.IGNORECASE)
    except re.error:
        return sentence
    return regex.sub(mask, sentence)


def find_occurrences(sentences, query, whole_word=True, max_results=300):
    """Every sentence containing `query` (case-insensitive), in reading
    order — a concordance: seeing every real use of a word or phrase
    across a whole book, not just its first, is what a close reader or
    researcher studying how a term is actually used needs.
    build_example_sentences() (one example per glossary word) only ever
    surfaces the first occurrence; this surfaces all of them, and isn't
    limited to words already in the glossary — any word or phrase can be
    searched, including proper nouns and phrases with no dictionary
    definition at all.

    `sentences` is a pre-split list from split_sentences() — pass the
    same cached list build_example_sentences() used rather than
    re-splitting the whole book on every search.

    `whole_word=True` (the default) matches `query` only at word
    boundaries, so searching "cat" won't also match "category" or "cats"
    — set False for a plain substring match instead, to cast a wider net
    (e.g. a search root like "cat" surfacing "cats" too).

    Every real match is returned, regardless of how long its sentence
    is. An earlier version skipped sentences over ~350 characters, on
    the theory that they're usually PDF-extraction artifacts (merged
    paragraphs, tables) rather than real prose — but that meant a word
    whose only occurrences happened to sit in genuinely long (real)
    sentences produced "No matches found" here even though the word
    plainly IS in the book, with no way to tell that apart from an
    actual bug. This is the one search feature explicitly meant to
    answer "does this word really appear, and where" — silently hiding
    real matches for a length heuristic undermines exactly that, so it
    doesn't happen here even at the cost of occasionally rendering an
    ugly, artifact-long "sentence" — a visibly odd result is still far
    more honest than a confident-looking empty one.

    Returns (matches, total): `matches` is a list of (sentence, spans) —
    the sentence text plus every match's (start, end) character span
    within it, for highlighting — capped at `max_results` so a very
    common word in a long book doesn't try to render thousands of rows.
    `total` is the true number of matching sentences, which can exceed
    len(matches) once the cap is hit.
    """
    query = query.strip()
    if not query:
        return [], 0
    try:
        regex = re.compile(_concordance_pattern(query, whole_word), re.IGNORECASE)
    except re.error:
        return [], 0
    matches = []
    total = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        spans = [m.span() for m in regex.finditer(sentence)]
        if not spans:
            continue
        total += 1
        if len(matches) < max_results:
            matches.append((sentence, spans))
    return matches, total


def detect_language(text):
    """Guess the ISO 639-1 language code of a book from a text sample.

    Detecting once from a large sample is far more reliable than
    detecting per word (very short words are too ambiguous on their own).
    Returns 'en' if detection fails, so callers default to the fast,
    fully-offline English path.
    """
    sample = text[:20000].strip()
    if not sample:
        return "en"
    try:
        return detect(sample)
    except LangDetectException:
        return "en"


def language_name(code):
    return LANGUAGE_NAMES.get(code, code)


def define(word, lang="eng"):
    """Return a WordNet definition for a word, or None if absent.

    `lang` is an NLTK WordNet language code (ISO 639-3) — "eng" (English,
    the default) or one of the many other languages NLTK's bundled Open
    Multilingual Wordnet data covers (confirmed by direct enumeration —
    `wn.langs()` after downloading omw-1.4 — this is a much longer list
    than once assumed here: "spa"/Spanish, "arb"/Arabic, and "fra"/French
    all have real native coverage, alongside ~30 others; conspicuously
    absent is Russian, which has none at all). See WORDNET_LANG_CODES for
    which of these this app actually exposes as a selectable language.

    For English, WordNet's own default sense ordering is used as-is,
    EXCEPT when the first-listed sense has zero attested corpus
    frequency — see _best_synset()'s docstring for why that specific,
    narrow case (and only that case) also falls back to the same
    frequency-weighted scoring every other language always uses. For
    other languages that ordering is
    NOT reliable on its own — verified directly: Spanish "libro" (book)
    picks "the third compartment of a ruminant's stomach" as its literal
    first listed sense, when the reader obviously means "book".

    Sense selection for non-English tried two simpler approaches first,
    both verified directly to fail differently:
      - Pure highest-tagged-corpus-frequency (an earlier version of this
        function): fixes "libro", but picks French "chat" (cat) as
        "guy.n.01" ("an informal term for a youth or man", freq=33) over
        the correct "cat.n.01" (freq=18) — "guy" simply happens to be
        tagged more often in an ENGLISH corpus, which has no bearing on
        which sense is the right French translation. Also wrong on
        French "table" (picks the spreadsheet sense over the furniture
        one) and "travail" (picks "the act of using" over "work").
      - Filter out zero-frequency senses, then trust OMW's own listed
        order: fixes "chat" and "table", but badly regresses common
        words where the correct sense is a bit further down the list but
        overwhelmingly more attested — Spanish "agua" (water) picks
        "a facility that provides a source of water" (freq=2, listed
        first) over the correct "water.n.01" (freq=136, listed 4th);
        "casa" (house) similarly picks a freq=2 sense over the correct
        one at freq=157.
    What's used instead: each sense is scored as
    (tagged-corpus frequency) / (1-based position in WordNet's own
    listed order), and the highest-scoring sense wins. This lets a big
    enough frequency gap still overrule an earlier list position (fixing
    "libro"/"agua"/"casa", where the correct sense is both far more
    frequent AND not first in line) while an earlier position can still
    overrule a small, coincidental frequency edge (fixing "chat", where
    a position-13-of-13 English slang sense would otherwise beat a
    position-4 correct one by a thin frequency margin). Verified
    directly across 12 common French/Spanish words: 11/12 correct: the
    one remaining miss, French "montagne" (mountain), resolves to "a
    local and well-defined elevation of the land" — WordNet's own "hill"
    sense — rather than the more specific "mountain.n.01"; a near-miss on
    a genuinely ambiguous scale word, not a wrong topic entirely. This is
    a genuinely hard problem (word-sense disambiguation, an actively
    researched area of NLP) with no clean closed-form solution from
    frequency and position alone — two more residual failure classes,
    confirmed directly and accepted rather than chased further given how
    much worse the two simpler alternatives above already are: short
    function words (articles, conjunctions, prepositions) frequently
    resolve to nonsense across EVERY non-English language this app
    supports, not just new ones — e.g. Spanish "de" resolves to
    "Delaware" — because WordNet fundamentally models content words, not
    grammatical ones, and no amount of frequency/position tuning fixes a
    category error like that; and occasionally a real content word still
    loses to an earlier-listed, moderately-frequent wrong sense despite a
    correct sense with much higher raw frequency further down the list
    (French "maison", house, resolves to an adverbial "homeward" sense
    at position 1/24, freq=59, over the correct "house.n.01" at position
    16/24, freq=157 — position's weight in the scoring formula still
    wins there). Both are pre-existing limits of this whole approach,
    not specific to any one language.

    A LookupError here (the multilingual data missing/not downloaded —
    confirmed this raises unhandled rather than returning no matches) is
    treated the same as "not found" rather than allowed to propagate and
    abort the whole glossary partway through over one word.
    """
    synset = _best_synset(word, lang)
    return synset.definition() if synset else None


def _sense_score(indexed_synset):
    position, synset = indexed_synset
    freq = sum(lemma.count() for lemma in synset.lemmas())
    return freq / position


def _best_synset(word, lang="eng"):
    """The single WordNet sense define() and word_pos() both derive
    their answer from — English uses WordNet's own first listed sense;
    every other language uses the frequency/position-weighted scoring
    described in define()'s docstring. Returns None if lookup fails or
    nothing matches. Both define() and word_pos() call this rather than
    each independently re-picking a sense, so they can never disagree
    with each other about which sense of an ambiguous word they mean.

    `lang` falsy (None — what WORDNET_LANG_CODES.get() returns for a
    language with no WordNet data at all, e.g. German) short-circuits
    before ever calling into NLTK: confirmed directly this matters, not
    just tidiness — build_glossary()'s own definition lookup already
    guards this same case with `if wordnet_lang:` before calling
    define(), but word_pos() (used independently, not just alongside
    define()) has no equivalent caller-side guard to rely on, and
    without one here `wn.synsets(word, lang=None)` raises a WordNetError
    — which, confirmed directly, is NOT a LookupError subclass despite
    living right next to LookupError-raising code in the same module, so
    the `except LookupError` below alone doesn't catch it; both are
    handled here so an unsupported language code (None, or any other
    string NLTK's omw-1.4 doesn't recognize) reliably means "no part of
    speech available" rather than an unhandled crash.
    """
    if not lang:
        return None
    try:
        synsets = wn.synsets(word, lang=lang)
    except (LookupError, WordNetError):
        return None
    if not synsets:
        return None
    if lang == "eng":
        first = synsets[0]
        # WordNet's own listed order is trustworthy for English EXCEPT
        # in one specific, checkable case: reported directly by a user
        # who spotted "recent" defined as "approximately the last
        # 10,000 years" — WordNet's noun sense for the Holocene epoch,
        # listed first ahead of the everyday adjective "recent" senses
        # purely by lexicographer-file ordering, with a tagged-corpus
        # frequency of exactly zero (i.e. never once attested as the
        # intended sense in the reference corpus these counts come
        # from). Confirmed by scanning all 8,680 words in cefr_en.json:
        # ~1,750 have a freq-0 first sense, and among those, sampling
        # broadly across the ~690 that a frequency-based re-pick would
        # actually change turns up dozens more of the exact same
        # failure shape ("am" as the element Americium, "add" as
        # Attention Deficit Disorder, "grey" as the novelist Zane Grey,
        # "heavy" as "an actor who plays villainous roles", "altogether"
        # as "informal term for nakedness") against no clear regressions
        # — a freq-0 first sense reliably means "never actually used
        # this way", so falling back to the same frequency/position
        # scoring already used for every other language is safe here.
        # Deliberately narrow: only overrides when position 1 itself
        # has zero attestation — a first sense with ANY real frequency,
        # however small, is left alone, since that case is where
        # WordNet's default ordering is actually backed by evidence
        # (and where a same-headword-different-part-of-speech reading
        # legitimately being more frequent overall, e.g. "lead" the
        # verb outscoring "lead" the noun, isn't grounds to override a
        # specific, already-attested sense).
        if sum(lemma.count() for lemma in first.lemmas()) == 0:
            best = max(enumerate(synsets, start=1), key=_sense_score)[1]
            if sum(lemma.count() for lemma in best.lemmas()) > 0:
                return best
        return first
    return max(enumerate(synsets, start=1), key=_sense_score)[1]


# WordNet's own single-letter POS tags, expanded to what a reader
# actually wants to see. "s" (adjective satellite — a sense that only
# exists tethered to a head adjective, e.g. "cardiac" relates to
# "heart") reads to a reader exactly like a plain adjective, so both
# fold into the same label rather than exposing a WordNet-internal
# distinction nobody outside computational linguistics cares about.
_WORDNET_POS_NAMES = {
    wn.NOUN: "noun", wn.VERB: "verb", wn.ADJ: "adjective", wn.ADJ_SAT: "adjective", wn.ADV: "adverb",
}


def word_pos(word, lang="eng"):
    """The part of speech (noun/verb/adjective/adverb) of the same
    WordNet sense define() would pick for `word` — or None if WordNet
    has no entry for it at all, which includes every word from a purely
    offline-dictionary-backed language (German, Turkish, Russian, and
    most Arabic — none of dict_de.sqlite3/dict_tr.sqlite3/
    dict_ar.sqlite3/dict_ru.sqlite3 carry part-of-speech data; see
    OFFLINE_DICT_LANGS above). Confirmed directly: those four bundled
    dictionaries' `definition` column is free text only, no tagging of
    any kind — adding real POS data for them would mean rebuilding each
    from its source with that field captured, not something this
    function can produce from what's already on disk.
    """
    synset = _best_synset(word, lang)
    if synset is None:
        return None
    return _WORDNET_POS_NAMES.get(synset.pos())


def build_pos_map(defined_words, wordnet_lang="eng"):
    """{word: part-of-speech} for every word in `defined_words` that
    WordNet has an entry for — see word_pos() for what "part of speech"
    means and which words end up simply absent from the returned dict
    rather than mapped to None (every offline-dictionary-only language,
    plus any WordNet-backed word WordNet itself has no entry for, e.g. a
    word an offline dictionary defined but WordNet doesn't cover).

    `defined_words` should be just the words build_glossary() actually
    found a definition for (its `defined` list's words, not the full
    extracted word list) — a word with no definition at all almost
    certainly has no WordNet entry either, so looking those up too would
    just be wasted lookups.

    `wordnet_lang` falsy (a language with no WordNet data at all, e.g.
    German) returns {} immediately without looking anything up — same
    short-circuit as _best_synset(), just skipping the per-word loop
    entirely instead of relying on each call to no-op.
    """
    if not wordnet_lang:
        return {}
    pos_map = {}
    for word in defined_words:
        pos = word_pos(word, wordnet_lang)
        if pos is not None:
            pos_map[word] = pos
    return pos_map


def word_relations(word, lang="eng"):
    """{"synonyms", "antonyms", "hypernyms", "hyponyms", "meronyms",
    "holonyms"}, each a list of real words from WordNet — lexical
    semantics, i.e. how this word relates to others in meaning, not just
    its own definition. Returns None if WordNet has no entry at all
    (same languages/limits as word_pos()/define()).

    Uses the exact same sense _best_synset() picks for define()/
    word_pos() — not every sense of a polysemous word — so these
    relations always describe the one sense the reader's definition and
    example are already showing them, never a different, unrelated
    sense of the same spelling (the same discipline build_cefr_map()'s
    strict-mode fallback enforces for CEFR level, applied here too:
    showing "bank" the riverside's hyponyms when the definition on
    screen is "bank" the financial institution would be actively
    misleading, not just imprecise).

    Every entry is real, curated WordNet data — no invented or
    guessed relation, and no synonym/antonym list assembled from
    co-occurrence statistics or any other approximation. `word` itself
    is excluded from its own synonym list (a synset's lemma set always
    includes the query word, which isn't informative shown back to the
    reader as if it were a distinct synonym). Meronyms/holonyms merge
    WordNet's three sub-kinds (part/substance/member) into one list each
    — a reader wants "what this is made of/part of" as one concept, not
    three separately-labeled near-empty lists for most words.
    """
    synset = _best_synset(word, lang)
    if synset is None:
        return None

    def _names(synsets):
        seen = []
        for s in synsets:
            name = s.lemmas()[0].name().replace("_", " ")
            if name not in seen:
                seen.append(name)
        return seen

    synonyms = []
    antonyms = []
    for lemma in synset.lemmas():
        name = lemma.name().replace("_", " ")
        if name.lower() != word.lower() and name not in synonyms:
            synonyms.append(name)
        for ant in lemma.antonyms():
            ant_name = ant.name().replace("_", " ")
            if ant_name not in antonyms:
                antonyms.append(ant_name)

    return {
        "synonyms": synonyms,
        "antonyms": antonyms,
        "hypernyms": _names(synset.hypernyms()),
        "hyponyms": _names(synset.hyponyms()),
        "meronyms": _names(synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms()),
        "holonyms": _names(synset.part_holonyms() + synset.substance_holonyms() + synset.member_holonyms()),
    }


_ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
# The reduced/unstressed reading of a vowel that carries CMU's "0"
# (no stress) marker — standard English phonetics (e.g. the schwa in
# the second syllable of "sofa"), not a simplification specific to
# this app.
_ARPABET_UNSTRESSED_IPA = {"AH": "ə", "ER": "ɚ"}
_ARPABET_PHONEME_RE = re.compile(r"^([A-Z]+)([012])?$")


def ensure_linguistics_data():
    """Make sure the CMU Pronouncing Dictionary and NLTK's stopwords
    corpus (every language in _STOPWORDS_LANGS, English included — it's
    one downloaded package, not a per-language fetch) are available
    before use — same try-real-call/except-LookupError/nltk.download()
    pattern as ensure_wordnet(), for the data word_pronunciation()/
    find_collocations() need. Called lazily, on first use of either (see
    app.py's WordInfoWindow opener), not during the main glossary
    pipeline — unlike WordNet/the POS tagger, nothing on the main
    word-table path needs these.
    """
    try:
        cmudict.dict()
    except LookupError:
        nltk.download("cmudict")
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def word_pronunciation(word, lang="en"):
    """{"ipa": str, "syllables": int} for one pronunciation of `word`, or
    None if it can't be transcribed.

    `lang="es"` dispatches to spanish_pronunciation() below — a
    deterministic rule-based transcription rather than a dictionary
    lookup, since (unlike English) that's the CORRECT tool for Spanish:
    see that function's own docstring for why. Every other `lang` value
    uses the English CMU-dictionary path below unchanged (this
    parameter is purely additive — see word_info.py's caller, the only
    one that ever needed to distinguish).

    {"ipa": str, "syllables": int} for one American-English
    pronunciation of `word`, from the CMU Pronouncing Dictionary
    (35,000+ words, hand-curated for decades of speech-recognition
    research — a real, precise phonemic transcription, not a rule-based
    guess at English's notoriously irregular spelling-to-sound mapping)
    — or None if the word isn't in it at all (proper nouns, rare/
    technical vocabulary, and any non-English word all fall in this
    bucket, same ceiling as this app's other "real dataset, not
    exhaustive" features).

    A word with several listed pronunciations (e.g. "reading" — the
    short-e past-tense-of-"read" reading vs. the long-e present-
    participle one) uses whichever CMU lists first — confirmed directly
    that this is NOT reliably "the most common one" (CMU's own ordering
    isn't frequency-ranked), so this picks a real, valid pronunciation
    of the word, not necessarily the one a given sentence is using.
    Precise about what it shows, not precise about picking the "right"
    sense from context — same honest boundary as define()'s single-
    best-sense choice for a genuinely ambiguous word.

    IPA is assembled phoneme-by-phoneme from CMU's ARPABET transcription
    via the standard ARPABET-to-IPA correspondence (_ARPABET_TO_IPA — a
    fixed, well-documented mapping table, not invented here), with
    primary/secondary stress marked (ˈ/ˌ) before the vowel that carries
    it. Syllable count is exact, not estimated: CMU marks every vowel
    phoneme with a stress digit, so counting those digits IS the
    syllable count, unlike _estimate_syllables() below (used only where
    no real pronunciation data exists at all).
    """
    if lang == "es":
        return spanish_pronunciation(word)
    try:
        entries = cmudict.dict().get(word.lower())
    except LookupError:
        return None
    if not entries:
        return None
    ipa_parts = []
    syllables = 0
    for phoneme in entries[0]:
        match = _ARPABET_PHONEME_RE.match(phoneme)
        if not match:
            continue
        base, stress = match.group(1), match.group(2)
        if stress is not None:
            syllables += 1
            if stress == "1":
                ipa_parts.append("ˈ")
            elif stress == "2":
                ipa_parts.append("ˌ")
            if stress == "0" and base in _ARPABET_UNSTRESSED_IPA:
                ipa_parts.append(_ARPABET_UNSTRESSED_IPA[base])
                continue
        ipa_parts.append(_ARPABET_TO_IPA.get(base, base))
    return {"ipa": "".join(ipa_parts), "syllables": syllables}


# ---------------------------------------------------------- Spanish phonology
#
# Unlike English (word_pronunciation() above), Spanish spelling maps to
# sound almost perfectly regularly — this is real, textbook Spanish
# phonics/phonology (the same rules taught to learners and used by
# published Spanish grapheme-to-phoneme systems), not a guess or an
# approximation invented for this app. A rule-based transcriber is
# therefore the CORRECT tool here, not a lesser substitute for a
# dictionary the way it would be for English — there is no equivalent
# of CMU's dictionary needed (or, for that matter, available) because
# Spanish orthography doesn't have English's kind of irregularity for
# one to resolve.
#
# Two deliberate, labeled dialect choices (Spanish has real regional
# pronunciation variation, so a single broad transcription has to pick
# one norm — same as word_pronunciation() above already deliberately
# picking American, not British, English):
#   - seseo: c (before e/i) and z are transcribed /s/, not the
#     Peninsular Spanish interdental /θ/. Seseo is how the large
#     majority of the world's Spanish speakers (all of Latin America,
#     plus parts of Spain) actually pronounce these letters.
#   - yeísmo: ll and y (as a consonant) are both transcribed /ʝ/, not
#     kept distinct (/ʎ/ for ll) the way some Peninsular/Andean dialects
#     still do. Yeísmo is likewise the majority pronunciation today.
# Both are named and called out here, and in the UI (WordInfoWindow),
# rather than silently presented as "the" pronunciation of Spanish.
#
# Also broad/phonemic, not narrow/allophonic — same granularity choice
# already made for English above (e.g. it does not attempt b/d/g's real
# spirantized [β/ð/ɣ] pronunciation between vowels, or nasal place
# assimilation), and does not model informal/rapid-speech reductions
# (e.g. syllable-final /s/ weakening) real speech often has. A precise
# citation of the FORMAL, standard pronunciation, same honest scope as
# the rest of this app's linguistic features.
_SPANISH_STRONG_VOWELS = set("aeo")
_SPANISH_WEAK_VOWELS = set("iu")
# Bare vowel + (is a written stress accent, not just a diaeresis). "ü"
# is the diaeresis mark (only ever appears in "güe"/"güi", to say "yes,
# pronounce the u here" — see _spanish_consonant_ipa()) — a real
# accent mark, in the sense of a mandatory syllabic-stress diacritic, it
# is not, so it's deliberately excluded from _SPANISH_ACCENTED.
_SPANISH_VOWEL_MAP = {
    "a": ("a", False), "e": ("e", False), "i": ("i", False), "o": ("o", False), "u": ("u", False),
    "á": ("a", True), "é": ("e", True), "í": ("i", True), "ó": ("o", True), "ú": ("u", True),
    "ü": ("u", False),
}
_SPANISH_ACCENTED = set("áéíóú")
# Two-letter Spanish consonant digraphs that are ONE sound (and,
# crucially, ONE syllabification unit — "carro" is "ca-rro", not
# "car-ro"; "mucho" is "mu-cho", not "muc-ho"). Checked for at every
# consonant-run position before falling back to single-letter mapping.
_SPANISH_DIGRAPH_IPA = {"ch": "tʃ", "ll": "ʝ", "rr": "r"}
# Consonant + l/r onset clusters that likewise never split across a
# syllable boundary ("templo" is "tem-plo", not "temp-lo"; "abrir" is
# "a-brir", not "ab-rir") — standard Spanish syllabification, the same
# "obstruent + liquid stays together" rule most languages with
# consonant clusters share.
_SPANISH_INSEPARABLE_CLUSTERS = {
    "pl", "bl", "cl", "gl", "fl", "pr", "br", "cr", "gr", "fr", "dr", "tr",
}


def _spanish_consonant_ipa(letters, i):
    """(ipa, letters_consumed) for the consonant unit starting at index
    `i` in the lowercase letter list `letters` — handles digraphs and
    the "qu"/"gu"/"gü" silent-or-pronounced-u spellings (which need to
    look at what follows before the mapping is known) before falling
    back to a fixed single-letter table. Returns (None, 1) for "h",
    which is written but never pronounced outside the "ch" digraph
    already handled above this call.
    """
    two = "".join(letters[i:i + 2])
    if two in _SPANISH_DIGRAPH_IPA:
        return _SPANISH_DIGRAPH_IPA[two], 2
    nxt = letters[i + 1] if i + 1 < len(letters) else ""
    if two == "qu" and nxt in ("e", "i", "é", "í"):
        return "k", 2
    if two == "gu" and nxt in ("e", "i", "é", "í"):
        return "g", 2
    if two == "gü":
        return "g", 2  # the ü itself is re-emitted as a normal weak vowel right after — see the tokenizer below
    ch = letters[i]
    if ch == "c":
        return ("s" if nxt in ("e", "i", "é", "í") else "k"), 1
    if ch == "g":
        return ("x" if nxt in ("e", "i", "é", "í") else "g"), 1
    if ch == "z":
        return "s", 1
    if ch in ("v", "b"):
        return "b", 1
    if ch == "j":
        return "x", 1
    if ch == "h":
        return None, 1
    if ch == "x":
        return "ks", 1
    if ch == "ñ":
        return "ɲ", 1
    if ch == "q":
        return "k", 1  # bare "q" with no "u" essentially never occurs in real words; a safe fallback, not a guessed rule
    if ch == "y":
        return "ʝ", 1
    if ch == "w":
        return "w", 1
    if ch == "r":
        # A single "r" is a genuinely different phoneme depending on
        # position — the real, minimal-pair distinction "pero" (tap,
        # /peɾo/, "but") vs. "perro" (trill, /pero/, "dog") turns on:
        # doubling it ("rr", handled above as its own digraph) always
        # trills, and so — even spelled with just one r — does starting
        # a word, or following "l"/"n"/"s" within one word ("Enrique",
        # "alrededor", "Israel" all trill there). Every other single-r
        # position (intervocalic, or after any other consonant, or
        # word-final) is the tap.
        prev = letters[i - 1] if i > 0 else None
        return ("r" if prev is None or prev in ("l", "n", "s") else "ɾ"), 1
    # p/t/d/k/f/s/l/m/n (and anything else stray, e.g. a foreign
    # loanword letter) are already their own correct IPA symbol as-is.
    return ch, 1


def _spanish_tokenize(word):
    """word -> [("C", ipa_or_None), ...] / [("V", bare_vowel, is_strong,
    is_accented), ...], one entry per letter position (a 2-letter
    digraph/qu/gu still yields exactly one "C" entry) — the raw material
    _spanish_syllabify() groups into syllables. A silent "u" in
    "qu"/"gu"+e/i is consumed by the consonant unit and never becomes
    its own "V" entry; a pronounced "ü" in "gü"+e/i (the diaeresis'
    entire purpose) is re-emitted right after "g" as an ordinary weak
    vowel "u", since that's exactly what it phonetically is once the
    diaeresis has done its one job of overriding "gu"'s usual silent-u
    spelling.
    """
    letters = list(word.lower())
    tokens = []
    i = 0
    n = len(letters)
    while i < n:
        ch = letters[i]
        if ch == "ü":
            tokens.append(("C", "g") if (i > 0 and letters[i - 1] == "g") else ("V", "u", False, False))
            i += 1
            continue
        if ch in _SPANISH_VOWEL_MAP:
            bare, accented = _SPANISH_VOWEL_MAP[ch]
            # A lone "y" acting as a vowel (the word "y" itself, meaning
            # "and", and a word-FINAL "y" after a vowel, e.g. "hoy",
            # "rey", "muy" — Spanish spelling uses "y" instead of "i" in
            # exactly these two positions) is handled in the "y"
            # consonant branch below by peeking backward/forward; this
            # branch only ever sees real vowel letters (a/e/i/o/u and
            # their accented forms), so nothing extra to do here.
            tokens.append(("V", bare, bare in _SPANISH_STRONG_VOWELS, accented))
            i += 1
            continue
        if ch == "y":
            is_word_and = word.lower() == "y"
            is_final_glide = (i == n - 1) and i > 0 and letters[i - 1] in _SPANISH_VOWEL_MAP
            if is_word_and or is_final_glide:
                tokens.append(("V", "i", False, False))
                i += 1
                continue
        ipa, consumed = _spanish_consonant_ipa(letters, i)
        tokens.append(("C", ipa))
        i += consumed
    return tokens


def _spanish_vowels_combine(v1, v2):
    """Whether two adjacent vowel tokens (each ("V", bare, is_strong,
    is_accented)) belong in the SAME syllable nucleus (a diphthong) or
    force a syllable break (hiatus) — the two real Spanish rules:
    two STRONG vowels together always hiatus (e.g. "po-e-ta"), and an
    ACCENTED weak vowel next to any vowel also always hiatus (that
    written accent's entire job, orthographically, is to mark exactly
    this — e.g. "dí-a", "pa-ís", "rí-o"). Everything else (weak+strong,
    strong+weak, unaccented weak+weak) combines into one syllable.
    """
    _, _, strong1, accented1 = v1
    _, _, strong2, accented2 = v2
    if strong1 and strong2:
        return False
    if (accented1 and not strong1) or (accented2 and not strong2):
        return False
    return True


def _spanish_syllabify(tokens):
    """[[(kind, ...), ...], ...] — `tokens` (from _spanish_tokenize())
    grouped into syllables, each a list of its own consonant/vowel
    tokens in original order. Two passes: group consecutive vowel
    tokens into nuclei (diphthongs/triphthongs vs. hiatus, via
    _spanish_vowels_combine()), then distribute each consonant run
    between the syllable before it and the syllable after it using
    standard Spanish onset-maximization — a single intervocalic
    consonant always joins the FOLLOWING syllable ("ca-sa"), a cluster
    of two joins the following syllable whole only if it's a real
    Spanish inseparable onset ("a-brir") and otherwise splits 1/1
    ("al-to"), and 3+ consonants peel off from the front into the
    preceding syllable's coda until what's left is a valid onset
    ("ins-truc-tor": "ns" peels off, "tr" is inseparable and stays).
    """
    nuclei = []
    consonant_runs = [[]]
    i = 0
    while i < len(tokens):
        if tokens[i][0] == "C":
            consonant_runs[-1].append(tokens[i][1])
            i += 1
            continue
        nucleus = [tokens[i]]
        i += 1
        while i < len(tokens) and tokens[i][0] == "V" and _spanish_vowels_combine(nucleus[-1], tokens[i]):
            nucleus.append(tokens[i])
            i += 1
        nuclei.append(nucleus)
        consonant_runs.append([])
    if not nuclei:
        return [[("C", c) for c in consonant_runs[0]]] if consonant_runs[0] else []

    syllables = [[] for _ in nuclei]
    # consonant_runs[0] = onset of the very first syllable (may be empty
    # for a vowel-initial word); consonant_runs[-1] = coda of the very
    # last syllable (may be empty for a vowel-final word); every run in
    # between sits BETWEEN two nuclei and needs splitting between them.
    syllables[0].extend(("C", c) for c in consonant_runs[0])
    syllables[0].extend(nuclei[0])
    for idx in range(1, len(nuclei)):
        cluster = consonant_runs[idx]
        # A silent "h" (see _spanish_consonant_ipa() — its IPA is
        # literally None, not a symbol) is never part of a real
        # consonant CLUSTER phonetically, since it has no sound to
        # cluster with — in native Spanish spelling it's always the
        # letter immediately before the vowel it "belongs" to, never
        # followed by another consonant. That assumption alone isn't
        # safe, though — confirmed directly against the bundled Spanish
        # dictionary's full ~749,000 headwords: transliterated proper
        # nouns and unassimilated loanwords ("Ahmadí", "Ahrimán", the
        # borrowed English "antiestablishment") DO put a consonant right
        # after an "h", so a fix that only special-cases "h" at the END
        # of a cluster (an earlier version of this code) still crashes
        # on those — confirmed directly this matters, not just in
        # theory: the string-based cluster checks below do
        # "".join(cluster), which raises outright the moment `None`
        # (from ANY position, not just last) reaches them. Since a
        # silent letter contributes nothing to the rendered IPA no
        # matter which syllable's consonant list it nominally ends up
        # in (spanish_pronunciation()'s render step already skips a
        # None wherever it occurs), it's safe to substitute "" for it
        # in this string-matching step alone — the substitution can
        # only ever affect which syllable a MUTE letter is filed under
        # internally, never the word's actual returned pronunciation.
        while len(cluster) > 2 or (
            len(cluster) == 2
            and "".join(c or "" for c in cluster) not in _SPANISH_INSEPARABLE_CLUSTERS
            and "".join(c or "" for c in cluster) not in _SPANISH_DIGRAPH_IPA
        ):
            syllables[idx - 1].append(("C", cluster[0]))
            cluster = cluster[1:]
        syllables[idx].extend(("C", c) for c in cluster)
        syllables[idx].extend(nuclei[idx])
    syllables[-1].extend(("C", c) for c in consonant_runs[-1])
    return syllables


def _spanish_stressed_syllable(word, syllables):
    """0-based index of the stressed syllable in `syllables` — a written
    accent (á/é/í/ó/ú) anywhere in the word always wins (standard
    Spanish spelling never marks more than one); otherwise the default
    rule every Spanish speaker learns in school: a word ending in a
    vowel, "n", or "s" stresses its next-to-last syllable, and a word
    ending in any other consonant stresses its last syllable.
    """
    for idx, syllable in enumerate(syllables):
        for token in syllable:
            if token[0] == "V" and token[3]:
                return idx
    last_letter = word[-1] if word else ""
    if last_letter in _SPANISH_STRONG_VOWELS or last_letter in _SPANISH_WEAK_VOWELS or last_letter in ("n", "s"):
        return max(0, len(syllables) - 2)
    return len(syllables) - 1


def _spanish_nucleus_ipa(nucleus):
    """IPA for one syllable's vowel nucleus (1-3 "V" tokens) — a single
    vowel is just its own phoneme; a diphthong/triphthong renders its
    weak vowel(s) as glides ("j"/"w" before the nucleus's peak, for a
    RISING diphthong like "tie-ne" -> "tje.ne"; the non-syllabic mark
    "i̯"/"u̯" after it, for a FALLING one like "ai-re" -> "ai̯.re") and
    its strong vowel as the syllable peak. A weak+weak pair (e.g. "ciu-
    dad") has no strong vowel to anchor on — real Spanish phonetics
    treats the SECOND vowel as the peak there, the first as a glide.
    """
    if len(nucleus) == 1:
        return nucleus[0][1]
    strong_positions = [i for i, v in enumerate(nucleus) if v[2]]
    peak = strong_positions[0] if strong_positions else len(nucleus) - 1
    parts = []
    for i, v in enumerate(nucleus):
        bare = v[1]
        if i == peak:
            parts.append(bare)
        else:
            glide = "j" if bare == "i" else "w"
            falling_glide = "i̯" if bare == "i" else "u̯"
            parts.append(glide if i < peak else falling_glide)
    return "".join(parts)


def spanish_pronunciation(word):
    """{"ipa": str, "syllables": int} for `word`, via the deterministic
    rule-based transcription described in this section's own module
    docstring above (seseo + yeísmo, broad/phonemic, formal register) —
    or None for an empty/non-alphabetic input. Unlike word_pronunciation()
    for English, this never returns None just because a word is rare or
    unlisted: the whole point of a rule-based approach is that it covers
    every correctly-spelled Spanish word equally, not just the ones a
    fixed dictionary happened to include.
    """
    word = word.lower().strip()
    if not word or not any(c.isalpha() for c in word):
        return None
    tokens = _spanish_tokenize(word)
    syllables = _spanish_syllabify(tokens)
    if not syllables:
        return None
    stressed = _spanish_stressed_syllable(word, syllables)
    # Each syllable (from _spanish_syllabify()) already lists its tokens
    # in real left-to-right spelling order — onset consonant(s), then
    # its vowel nucleus, then any CODA consonant(s) after the vowel
    # (e.g. "hom-bre"'s first syllable is onset "h"(silent) + nucleus
    # "o" + coda "m" — "om", not "mo"). So consonants emit immediately
    # as encountered, but a run of vowel tokens has to be collected and
    # rendered as ONE combined unit (glides resolved together) exactly
    # where it occurs in that order, not deferred to the syllable's end
    # (an earlier version of this function did that, which silently
    # reordered every syllable with a coda — caught by testing "hombre"
    # against its real, unambiguous pronunciation and finding "mobre").
    parts = []
    for idx, syllable in enumerate(syllables):
        if idx == stressed:
            parts.append("ˈ")
        j = 0
        while j < len(syllable):
            kind, *_rest = syllable[j]
            if kind == "C":
                if syllable[j][1] is not None:
                    parts.append(syllable[j][1])
                j += 1
                continue
            vowel_run = []
            while j < len(syllable) and syllable[j][0] == "V":
                vowel_run.append(syllable[j])
                j += 1
            parts.append(_spanish_nucleus_ipa(vowel_run))
    return {"ipa": "".join(parts), "syllables": len(syllables)}


def find_collocations(sentences, word, window=4, top_n=10, lang="en"):
    """The words that occur most often near `word` within THIS book's
    own sentences, most frequent first — real corpus-linguistics
    collocation extraction from the one corpus this app always has on
    hand (the loaded book itself) rather than a purchased/external
    corpus (COCA, BNC) this app has no access to. Counts any word within
    `window` tokens on either side of an occurrence of `word`, in the
    same sentence — collocations don't cross a sentence boundary, since
    that's not really "near" in any useful sense.

    `lang`'s own stopwords (the/a/of/and/... for English, el/la/de/y/...
    for Spanish) are excluded: without that, the top "collocations" of
    almost any word in almost any book are just the most frequent words
    in that language generally — technically correct, completely
    uninformative, and not what "collocation" means in actual corpus
    linguistics (content words next to the target word are the
    informative signal; function words are noise). Uses the same
    _stopword_set(lang) extract_phrases() already relies on for the same
    reason — already covers every language this app supports (see
    _STOPWORDS_LANGS), so this was never actually English-specific
    logic, just written that way before any caller needed otherwise.
    Degrades to no stopword filtering (still functions, just noisier)
    for a language _STOPWORDS_LANGS doesn't cover, or if the data isn't
    downloaded yet — same quiet fallback _stopword_set() itself already
    makes.

    Returns up to `top_n` (word, count) pairs. Ties are broken by first
    occurrence in the book, for a deterministic, reproducible order
    rather than depending on dict-iteration happenstance.
    """
    word = word.lower()
    stop_words = _stopword_set(lang)
    counts = {}
    order = []
    for sentence in sentences:
        tokens = [_strip_arabic_diacritics(t.lower()) for t in re.findall(_WORD_PATTERN, sentence, re.UNICODE)]
        for i, token in enumerate(tokens):
            if token != word:
                continue
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if j == i:
                    continue
                neighbor = tokens[j]
                if neighbor == word or neighbor in stop_words or len(neighbor) < 2:
                    continue
                if neighbor not in counts:
                    counts[neighbor] = 0
                    order.append(neighbor)
                counts[neighbor] += 1
    ranked = sorted(order, key=lambda w: (-counts[w], order.index(w)))
    return [(w, counts[w]) for w in ranked[:top_n]]


def _estimate_syllables(word):
    """Approximate syllable count via vowel-group counting — the same
    heuristic real readability tools (textstat and others) use for
    English, not a novel guess: count runs of consecutive vowels as one
    syllable each, then apply the two standard refinements — a trailing
    silent "e" doesn't count (except when it's the word's only vowel,
    e.g. "the"), and every real word has at least 1 syllable. Used only
    by readability_stats() below, which needs a syllable count for
    EVERY word in the whole book, not just the ~35,000 CMU covers —
    word_pronunciation()'s CMU-dictionary count is exact and always
    preferred where it's available (see its own docstring), but
    readability metrics need 100% coverage, so this fills the rest.
    """
    word = word.lower()
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)
    if count > 1 and word.endswith("e") and not word.endswith("le"):
        count -= 1
    return max(1, count)


_FLESCH_LEVELS = (
    (90, "Very Easy"), (80, "Easy"), (70, "Fairly Easy"), (60, "Standard"),
    (50, "Fairly Difficult"), (30, "Difficult"), (0, "Very Difficult"),
)


def _estimate_syllables_es(word):
    """Exact (not estimated, despite the name matching _estimate_syllables()
    above for a parallel call site) Spanish syllable count for `word`,
    via the same real syllabifier spanish_pronunciation() uses
    (_spanish_tokenize() + _spanish_syllabify()) — reused rather than a
    cruder vowel-group heuristic because it already exists and IS the
    linguistically correct answer, not an approximation of one. Falls
    back to 1 for a "word" with no letters at all (a stray digit/symbol
    _WORD_PATTERN still matched), same floor _estimate_syllables() uses.
    """
    syllables = _spanish_syllabify(_spanish_tokenize(word.lower()))
    return max(1, len(syllables))


def readability_stats(text, lang="en"):
    """{"avg_sentence_length", "lexical_density", "flesch_score",
    "flesch_level"} for this book's actual text, or None if there's not
    enough text to measure (no sentences/words at all) or `lang` isn't
    "en"/"es" — a real, published, language-specific readability formula
    exists for both (see below), calibrated for that language's own
    syllable/sentence patterns; there's no equivalent third formula
    (or reason to assume Flesch's or Fernández Huerta's own English/
    Spanish-specific coefficients would mean anything) for any other
    language this app supports, so those still get None, same
    "decline rather than guess" choice as estimate_vocabulary_level()
    staying English-only for the same underlying reason (no real
    CEFR-graded data for those languages either).

    English uses the classic Flesch Reading Ease formula (206.835 -
    1.015*(words/sentences) - 84.6*(syllables/words)). Spanish uses its
    real, standard adaptation — the Fernández Huerta formula (1959):
    206.84 - 60*(syllables/words) - 1.02*(words/sentences) — same shape,
    different coefficients, published specifically because Flesch's own
    English coefficients don't transfer (Spanish words average more
    syllables than English ones for a text of comparable difficulty, so
    using Flesch's coefficients on Spanish text would score every
    Spanish book as artificially "difficult"). Both land on the SAME
    0-100 scale with the SAME band cutoffs (confirmed directly: Fernández
    Huerta's own published difficulty bands — Muy fácil 90-100, Fácil
    80-90, Bastante fácil 70-80, Normal 60-70, Bastante difícil 50-60,
    Difícil 30-50, Muy difícil 0-30 — are numerically identical to
    _FLESCH_LEVELS' English ones), so the same table labels both; only
    the ENGLISH label text is shown either way, since the UI is English
    throughout and "Fairly Easy"/"Bastante fácil" are the same claim.

    `lexical_density` is unique words / total words * 100 — how
    repetitive vs. varied the vocabulary is (a children's book and a
    literary novel with the same word count read very differently here)
    — language-independent arithmetic, computed the same way for both.
    """
    if lang not in ("en", "es"):
        return None
    sentences = [s for s in split_sentences(text) if s.strip()]
    words = re.findall(_WORD_PATTERN, text, re.UNICODE)
    if not sentences or not words:
        return None
    total_words = len(words)
    syllable_counter = _estimate_syllables_es if lang == "es" else _estimate_syllables
    total_syllables = sum(syllable_counter(w) for w in words)
    unique_words = len({w.lower() for w in words})
    avg_sentence_length = total_words / len(sentences)
    avg_syllables_per_word = total_syllables / total_words
    if lang == "es":
        score = 206.84 - 60 * avg_syllables_per_word - 1.02 * avg_sentence_length
    else:
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    # default= _FLESCH_LEVELS[-1][1] ("Very Difficult"): both formulas
    # are unbounded below 0 for a genuinely dense enough text (long,
    # syllable-heavy words in short bursts — confirmed directly this
    # isn't just a theoretical edge case, a contrived but real example
    # of dense multi-syllabic vocabulary already drives English's score
    # negative), and _FLESCH_LEVELS' own lowest bucket starts AT 0, not
    # -inf, so a plain `next(...)` with no default raised StopIteration
    # on exactly that input — same bucket's label is still the right
    # one to show, just needs to actually catch the case its threshold
    # number alone doesn't.
    level = next((name for threshold, name in _FLESCH_LEVELS if score >= threshold), _FLESCH_LEVELS[-1][1])
    return {
        "avg_sentence_length": round(avg_sentence_length, 1),
        "lexical_density": round(unique_words / total_words * 100, 1),
        "flesch_score": round(score, 1),
        "flesch_level": level,
    }


# This app's ISO 639-1 codes -> the fileid NLTK's stopwords corpus
# lists it under. Confirmed directly: NLTK's stopwords corpus happens
# to cover every language this app already supports (unlike CMU
# pronunciation/Flesch readability, which really are English-only),
# so phrase extraction's stopword filter works for all of them, not
# just English — checked with stopwords.fileids() against
# OFFLINE_DICT_LANGS/WORDNET_LANG_CODES's own language list.
_STOPWORDS_LANGS = {
    "en": "english", "de": "german", "tr": "turkish", "ru": "russian",
    "ar": "arabic", "fr": "french", "es": "spanish",
}


def _stopword_set(lang):
    """The stopword set for `lang` (an ISO 639-1 code), or an empty set
    if `lang` isn't one _STOPWORDS_LANGS covers or the data isn't
    downloaded — callers degrade to "no stopword filtering" rather than
    erroring, same quiet-degrade choice as _load_cefr_data().
    """
    fileid = _STOPWORDS_LANGS.get(lang)
    if not fileid:
        return set()
    try:
        return set(stopwords.words(fileid))
    except LookupError:
        return set()


def ensure_phrase_data(lang="en"):
    """Make sure the stopword list phrase extraction uses for `lang` is
    downloaded — same try/except-LookupError/nltk.download() shape as
    ensure_wordnet()/ensure_linguistics_data(). A `lang` with no
    _STOPWORDS_LANGS entry is a no-op (extract_phrases() just runs
    without stopword filtering for it, per _stopword_set()).
    """
    fileid = _STOPWORDS_LANGS.get(lang)
    if not fileid:
        return
    try:
        stopwords.words(fileid)
    except LookupError:
        nltk.download("stopwords")


def _phrase_meaning(gram, wordnet_lang):
    """A real WordNet definition for `gram` (a tuple of words, in
    whatever inflected form they actually appeared in the book), if
    WordNet catalogues this phrase as its own lexical entry — see
    extract_phrases()'s docstring for what that does and doesn't cover.

    Tries, in order: (1) the phrase exactly as it appeared ("looked
    after"); (2) with its first word reduced to its base form via
    word_family_root() ("look after") — WordNet's own multi-word
    entries are indexed under their base form ("look_after"), not every
    inflection, and the leading word is where a phrasal verb's tense
    normally shows up ("look/looked/looking after", "give/gave/given
    up"); (3) with EVERY word reduced individually, for the rarer case
    where a later word carries the relevant inflection. Confirmed
    directly this matters, not just theoretically: "looked after" (the
    literal past-tense text) has no WordNet entry on its own, while
    "look after" does — without this fallback, a real, correctly-
    catalogued idiom would show no meaning purely because the book
    happened to use it in past tense. Returns None if nothing at any
    step resolves to a real WordNet entry — never a guessed meaning.
    """
    candidates = [gram]
    lemma0 = word_family_root(gram[0], wordnet_lang) if wordnet_lang == "eng" else None
    if lemma0:
        candidates.append((lemma0,) + gram[1:])
    if wordnet_lang == "eng":
        all_lemmas = tuple(word_family_root(w, wordnet_lang) or w for w in gram)
        if all_lemmas not in candidates:
            candidates.append(all_lemmas)
    for candidate in candidates:
        meaning = define("_".join(candidate), wordnet_lang)
        if meaning:
            return meaning
    return None


_PHRASE_DICTIONARY_CACHE = {}


_PHRASE_DICTIONARY_FILES = {"en": "phrase_en.json", "es": "phrase_es.json"}


def _load_phrase_dictionary(lang="en"):
    """{phrase: {pos: {"definition", "idiomatic"}}} from the bundled
    phrase_en.json/phrase_es.json — real, editorially-catalogued idioms,
    phrasal verbs, and fixed expressions for `lang`, built from
    Wiktionary's own lang_code-filtered multi-word entries (CC BY-SA 4.0
    + GFDL; see phrase_data/README.md for the exact source and build
    script — same "source/build material kept separately, only the
    derived resource ships" split as cefr_en.json). Returns {} for any
    `lang` with no entry in _PHRASE_DICTIONARY_FILES (no bundled data at
    all for that language yet). Cached per-language after first load,
    same quiet-degrade-to-{} choice as _load_cefr_data() if the file is
    missing/corrupt.
    """
    filename = _PHRASE_DICTIONARY_FILES.get(lang)
    if not filename:
        return {}
    if lang not in _PHRASE_DICTIONARY_CACHE:
        path = os.path.join(_resource_dir(), filename)
        try:
            with open(path, encoding="utf-8") as f:
                _PHRASE_DICTIONARY_CACHE[lang] = json.load(f)
        except (OSError, ValueError):
            _PHRASE_DICTIONARY_CACHE[lang] = {}
    return _PHRASE_DICTIONARY_CACHE[lang]


def phrase_dictionary_lookup(gram, lang="en"):
    """(canonical_phrase, pos, definition, idiomatic) for `gram` (a
    tuple of words, in whatever inflected form they actually appeared
    in the book), if this language's bundled phrase dictionary
    catalogues it as a real entry — or None if it doesn't.

    For English, tries the exact wording first, then the same lemma-
    fallback candidates _phrase_meaning() uses (base form of the first
    word; base form of every word) — necessary because Wiktionary
    indexes a phrasal verb under its base form ("give up"), not every
    inflection ("gave up", "given up", "giving up"), so a book using
    the inflected form would otherwise never match its own dictionary
    entry. For Spanish (or any future language), this fallback is
    skipped: word_family_root() is English-morphy-only (see its own
    docstring) and has no Spanish equivalent bundled here, so only
    EXACT spelling matches against phrase_es.json's headwords are
    found. A real, stated limitation, not a silent gap — see
    phrase_data/README.md: many Spanish idioms are wholly invariant
    ("sin embargo", "de vez en cuando") and match exactly regardless,
    but one with a conjugating verb ("tomar el pelo") only matches its
    bare infinitive spelling, not "le tomó el pelo".

    `canonical_phrase` in the return value is the dictionary's own
    headword spelling (space-joined) — NOT necessarily the literal text
    from the book — since every inflected surface form of one phrasal
    verb should count as, and be reported as, the SAME phrase, not one
    entry per inflection.

    When a phrase has entries under more than one part of speech (e.g.
    "give up" as both verb and adjective in Wiktionary), a tagged-
    idiomatic sense wins first (a real editorial judgment call worth
    surfacing over a plainer sense), otherwise "verb" is preferred
    (phrasal verbs are the single most useful category for a language
    learner), otherwise whichever pos happens to come first — there's
    no book-context part-of-speech available to disambiguate by, unlike
    single-word CEFR lookups which have build_pos_map() to lean on.
    """
    candidates = [gram]
    if lang == "en":
        lemma0 = word_family_root(gram[0], "eng")
        if lemma0:
            candidates.append((lemma0,) + gram[1:])
        all_lemmas = tuple(word_family_root(w, "eng") or w for w in gram)
        if all_lemmas not in candidates:
            candidates.append(all_lemmas)

    dictionary = _load_phrase_dictionary(lang)
    for candidate in candidates:
        by_pos = dictionary.get(" ".join(candidate))
        if not by_pos:
            continue
        idiomatic_pos = next((p for p, d in by_pos.items() if d.get("idiomatic")), None)
        pos = idiomatic_pos or ("verb" if "verb" in by_pos else next(iter(by_pos)))
        data = by_pos[pos]
        return " ".join(candidate), pos, data["definition"], bool(data.get("idiomatic"))
    return None


def extract_phrases(sentences, lang="en", min_freq=2, max_ngram=4, top_n_per_length=150):
    """Multi-word phraseological units actually used in this book, in
    two clearly separate tiers.

    "recognized" — REAL idioms, phrasal verbs, and fixed expressions:
    every 2-to-`max_ngram`-word span in the book is checked against
    phrase_dictionary_lookup()'s bundled Wiktionary-derived catalogue
    (see its own docstring), and every span that matches a real entry
    is kept, however many times it occurs — even once, unlike
    "recurring" below, since a single occurrence of a genuine,
    dictionary-verified idiom is already good evidence on its own; there's
    no statistical threshold to clear when the dictionary itself already
    is the evidence. This tier is the actual fix for what an earlier,
    purely-statistical version of this function could never do: that
    version could only ever say "these words co-occur often in this
    book," never "this is a real, recognized phrase."

    "recurring" — the ORIGINAL statistical approach, kept as a second,
    clearly separate tier for genuinely recurring word combinations
    that AREN'T in the dictionary (a book-specific turn of phrase, a
    running motif, a name treated as a fixed unit, etc. — still worth
    seeing, just not claimed to be an established idiom). Counted per
    length, kept only at `min_freq`+ occurrences, excluding all-
    stopword spans and anything the "recognized" tier already claimed
    (by its surface form), bigrams ranked by pointwise mutual
    information (PMI) — log2(P(w1,w2) / (P(w1)*P(w2))), the standard
    corpus-linguistics measure of how much more often two words co-occur
    than their individual frequencies alone would predict by chance
    (this is what separates a real fixed combination from merely-
    frequent function-word glue like "of the", which raw frequency
    alone can't tell apart). Each entry here also carries a "meaning":
    a WordNet definition when _phrase_meaning() finds one (a smaller,
    secondary source, separate from the bundled phrase dictionary,
    already covering the "recognized" tier) — or None, the expected
    outcome for most recurring-but-uncatalogued combinations.

    Returns {"recognized": [{"phrase", "pos", "definition", "idiomatic",
    "count", "example"}, ...], "recurring": {length: {"entries": [...],
    "total_found": int}, ...}}. "recognized" is a flat list (its
    idiomatic/pos framing matters more than raw word count, unlike
    "recurring"'s length-based grouping), sorted idiomatic-first then by
    occurrence count; "recurring" lists are each capped at
    `top_n_per_length` with a `total_found` alongside for a "showing top
    150 of 640" caller message. The "recognized" tier runs for every
    `lang` _PHRASE_DICTIONARY_FILES has bundled data for (English and
    Spanish today — see phrase_dictionary_lookup()'s docstring for
    Spanish's one real limitation, exact-spelling-only matching) and is
    simply skipped (same as if the book had none) for any other
    language; "recurring" runs for every language this function always
    supported, regardless.
    """
    import math

    wordnet_lang = WORDNET_LANG_CODES.get(lang)
    stop_words = _stopword_set(lang)
    unigram_counts = {}
    ngram_counts = {n: {} for n in range(2, max_ngram + 1)}
    ngram_examples = {n: {} for n in range(2, max_ngram + 1)}
    total_unigrams = 0
    total_bigrams = 0
    recognized_by_key = {}     # canonical phrase -> accumulated entry

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = [_strip_arabic_diacritics(t.lower()) for t in re.findall(_WORD_PATTERN, sentence, re.UNICODE)]
        for tok in tokens:
            unigram_counts[tok] = unigram_counts.get(tok, 0) + 1
        total_unigrams += len(tokens)
        total_bigrams += max(0, len(tokens) - 1)

        # Every word position in THIS sentence a recognized dictionary
        # match has already claimed — checked longest-n-gram-first
        # below so a real, longer phrase ("once upon a time") wins over
        # a shorter span that's merely a fragment of it and might
        # ALSO, coincidentally, be its own dictionary entry ("upon a
        # time") or get double-counted as a "recurring" statistical
        # combination ("cats and dogs" inside "rain cats and dogs").
        # Confirmed directly this matters, not just theoretically: both
        # of those exact overlaps showed up as separate, redundant
        # entries before this check existed.
        claimed_positions = set()
        if lang in _PHRASE_DICTIONARY_FILES:
            for n in range(min(max_ngram, len(tokens)), 1, -1):
                for i in range(len(tokens) - n + 1):
                    span = range(i, i + n)
                    if claimed_positions.intersection(span):
                        continue
                    match = phrase_dictionary_lookup(tuple(tokens[i:i + n]), lang)
                    if not match:
                        continue
                    canonical, pos, definition, idiomatic = match
                    claimed_positions.update(span)
                    bucket = recognized_by_key.setdefault(canonical, {
                        "phrase": canonical, "pos": pos, "definition": definition,
                        "idiomatic": idiomatic, "count": 0, "example": sentence,
                    })
                    bucket["count"] += 1

        for n in range(2, max_ngram + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                if claimed_positions.intersection(range(i, i + n)):
                    continue
                gram = tuple(tokens[i:i + n])
                if stop_words and all(w in stop_words for w in gram):
                    continue
                ngram_counts[n][gram] = ngram_counts[n].get(gram, 0) + 1
                if gram not in ngram_examples[n]:
                    ngram_examples[n][gram] = sentence

    recognized = sorted(
        recognized_by_key.values(), key=lambda e: (e["idiomatic"], e["count"]), reverse=True,
    )

    recurring = {}
    for n in range(2, max_ngram + 1):
        qualifying = [
            (gram, count) for gram, count in ngram_counts[n].items() if count >= min_freq
        ]
        if not qualifying:
            continue
        entries = []
        for gram, count in qualifying:
            pmi = None
            if n == 2 and total_unigrams and total_bigrams:
                p_pair = count / total_bigrams
                p1 = unigram_counts[gram[0]] / total_unigrams
                p2 = unigram_counts[gram[1]] / total_unigrams
                if p1 and p2:
                    pmi = round(math.log2(p_pair / (p1 * p2)), 2)
            meaning = _phrase_meaning(gram, wordnet_lang) if wordnet_lang else None
            entries.append({
                "phrase": " ".join(gram), "count": count, "pmi": pmi,
                "example": ngram_examples[n][gram], "meaning": meaning,
            })
        entries.sort(key=lambda e: (e["pmi"] if e["pmi"] is not None else e["count"], e["count"]), reverse=True)
        recurring[n] = {"entries": entries[:top_n_per_length], "total_found": len(entries)}

    return {"recognized": recognized, "recurring": recurring}


_CEFR_DATA_FILES = {"en": "cefr_en.json", "es": "cefr_es.json"}
_CEFR_DATA_CACHE = {}


def _load_cefr_data(lang="en"):
    """{word: {pos: level}} from the bundled cefr_en.json/cefr_es.json
    (see cefr_data/README.md for what each is built from, and why no
    other language has one) for `lang`, cached per-language after first
    load since it's read-only for the life of the process. Returns {}
    immediately for any `lang` with no entry in _CEFR_DATA_FILES. A
    missing or corrupt file quietly returns {} too rather than
    erroring — same "degrade to no data" choice as load_progress() — so
    CEFR levels are simply absent everywhere else in the app instead of
    crashing it.
    """
    filename = _CEFR_DATA_FILES.get(lang)
    if not filename:
        return {}
    if lang not in _CEFR_DATA_CACHE:
        path = os.path.join(_resource_dir(), filename)
        try:
            with open(path, encoding="utf-8") as f:
                _CEFR_DATA_CACHE[lang] = json.load(f)
        except (OSError, ValueError):
            _CEFR_DATA_CACHE[lang] = {}
    return _CEFR_DATA_CACHE[lang]


_CEFR_LEVEL_RANK = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}


def cefr_level(word, pos=None, strict=False, lang="en"):
    """This word's CEFR level (one of "A1".."C2" for English; Spanish's
    bundled data — see cefr_data/README.md — tops out at C1, a real,
    documented gap in that source rather than this app's own doing, so
    "C2" simply never comes back for a Spanish word), or None if the
    bundled data for `lang` has no entry for it at all.

    Many words carry a different level per sense/part-of-speech (e.g.
    "above" is A1 as an adverb/preposition but B1 as an adjective, and
    "book" is A1 as a noun but B1 as a verb) — passing `pos` (the same
    string word_pos()/build_pos_map() produce, e.g. "noun") picks that
    sense's level when the data has it.

    When `pos` doesn't match any of `word`'s entries, the default
    (`strict=False`) falls back to the easiest (lowest) level across all
    of the word's known senses, as a benefit-of-the-doubt default — a
    reasonable choice when `word` really is the literal word being
    looked up (this data is keyed on it directly), since a mismatch here
    just means two taggers disagree about a nuance of the SAME word.

    `strict=True` returns None instead of falling back — for
    build_cefr_map()'s fallback tiers, which call this with a DIFFERENT
    word (a lemma or irregular-form root, not the word actually being
    looked up): there, a part-of-speech that doesn't match is a real
    signal the two are just a spelling coincidence (a homograph), not
    genuinely the same headword in another grammatical form. Confirmed
    directly: "crow" (the bird) is a noun here at B1, its only entry,
    and "crowed" used as a verb ("he crowed about his victory" — a
    different, unrelated word that just happens to inflect to the same
    root) was being shown as B1 too, silently borrowed from the bird
    sense with nothing marking it as a guess. A genuinely same-word
    inflection ("booked" -> "book", noun A1/verb B1) simply has the
    matching entry already, so strict mode was never needed for it —
    the borrowing this replaces only ever fired for exactly the
    homograph cases where it was actively misleading.
    """
    entries = _load_cefr_data(lang).get(word)
    if not entries:
        return None
    if pos:
        if pos in entries:
            return entries[pos]
        if strict:
            return None
    return min(entries.values(), key=lambda lvl: _CEFR_LEVEL_RANK[lvl])


_EXCEPTION_POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb"}


def _irregular_lemma_and_pos(word):
    """(lemma, cefr_pos_string) via WordNet's own bundled irregular-form
    exception lists (verb.exc/noun.exc/adj.exc/adv.exc — real curated
    linguistic data shipped inside the WordNet corpus itself, not a
    guess and not a new dependency), for exactly the case
    word_family_root()'s morphy lookup can't handle: an irregular form
    that ALSO happens to be a valid standalone word in its own right.
    Confirmed directly: "felt" is both the past tense of "feel" AND
    WordNet's own entry for the fabric (plus its own verb senses, "to
    felt" as in matting fibers together) — since "felt" is already a
    real headword, morphy("felt", VERB) returns "felt" unchanged rather
    than surfacing "feel" as an alternate irregular reading, even though
    wn.synsets("felt") clearly pulls in feel.v.01 etc. The exact mapping
    morphy silently drops is still sitting in WordNet's exception
    tables, which this reads directly instead. Returns (None, None) if
    nothing matches. `cefr_pos_string` (already in "noun"/"verb"/...
    form, matching _WORDNET_POS_NAMES's values) is the exception list
    `word` was actually found under — see build_cefr_map()'s docstring
    for why that's a more trustworthy signal than a separately-guessed
    part of speech for `word` itself.

    wn._exception_map is a private NLTK attribute (leading underscore,
    not part of its documented public API) — used anyway because it's
    the literal table wn.synsets()/wn.morphy() already consult
    internally, confirmed present and correctly keyed ("felt" -> "feel",
    "went" -> "go", "mice" -> "mouse", 2401 verb entries alone) on the
    nltk version this app ships against; a future nltk release renaming
    it would simply make this function a no-op again (caught below),
    not break anything else.
    """
    try:
        exception_map = wn._exception_map
    except AttributeError:
        return None, None
    for pos_key in ("n", "v", "a", "r"):
        lemmas = exception_map.get(pos_key, {}).get(word)
        if lemmas:
            return lemmas[0], _EXCEPTION_POS_NAMES[pos_key]
    return None, None


def build_cefr_map(defined_words, lang="en", pos_map=None):
    """{word: CEFR level} for every word in `defined_words` the bundled
    data covers — see cefr_level() for how a word's level is picked and
    cefr_data/README.md for each language's data's coverage/license.
    English and Spanish only (cefr_en.json/cefr_es.json have no other
    language's vocabulary), so any other `lang` short-circuits to {}
    without touching either data file at all, same shape as
    build_pos_map()'s `wordnet_lang` falsy short-circuit.

    `pos_map`, if given, should be the same build_pos_map() result
    already computed for this glossary — reused here (not recomputed)
    purely to disambiguate a level by sense; passing None just means
    every word falls back to its easiest sense's level.

    A word not found as-is (cefr_en.json only has ~8,700 headwords, so
    most inflected forms — "booked", "cats", "faster" — aren't literal
    entries) falls back, in order: (1) its base form via
    _morphy_lemma_and_pos() (regular inflections — "booked" -> "book"),
    then (2) its irregular-form lemma via _irregular_lemma_and_pos()
    (irregular verbs morphy alone won't rewrite because the inflected
    spelling is ALSO a real word — "felt" -> "feel"). Together these are
    the two biggest levers on coverage: confirmed directly that most of
    a real book's "no level shown" words are inflected forms of a
    headword the data already has, not genuinely missing vocabulary.
    Still not 100% — proper nouns, rare/technical words, and anything
    outside this dataset's ~8,700 headwords will still show no level,
    same as build_pos_map() never reaching 100% either. BOTH fallback
    tiers are English-only, skipped entirely for Spanish: morphy and
    WordNet's irregular-form exception tables are English-specific data
    (see word_family_root()'s and _irregular_lemma_and_pos()'s own
    docstrings) with no Spanish equivalent bundled here, so a Spanish
    word only ever gets an exact-spelling lookup against cefr_es.json —
    a real, stated limitation (an inflected Spanish form absent from
    that ~6,500-headword data shows no level), not a silent gap.

    Both fallback tiers check the root's level with `strict=True`,
    using the part of speech THE REDUCTION ITSELF SUCCEEDED UNDER — not
    `pos_map.get(word)` (the inflected word's own, separately-guessed
    part of speech). This distinction was confirmed necessary directly,
    not just theoretically: `word_pos()` picks the single "best"
    WordNet synset for a spelling in isolation, which for some irregular
    forms lands on a rare, unrelated sense — "sat" has its own obscure
    noun sense WordNet ranks highest, so word_pos("sat") says "noun",
    even though it's obviously the verb "sit" here. Gating strictness on
    that guess would have wrongly rejected "sit"'s real verb-A1 entry
    for a false-mismatch reason. The reduction's OWN successful category
    doesn't have this problem: morphy/the exception table found "sat"
    specifically by trying it as a VERB, so that's what gets checked
    against "sit"'s entries — and it's exactly this same category that
    correctly catches "crowed" (found as a VERB reduction of "crow",
    which only has a NOUN entry) as a genuine mismatch worth rejecting.
    """
    if lang not in _CEFR_DATA_FILES:
        return {}
    pos_map = pos_map or {}
    cefr_map = {}
    for word in defined_words:
        pos = pos_map.get(word)
        level = cefr_level(word, pos, lang=lang)
        # strict=True below: these two tiers hand back a DIFFERENT
        # word's data (a lemma/root), so — unlike the exact-match lookup
        # above — a part-of-speech mismatch here gets treated as
        # "probably a homograph, don't guess" rather than papered over.
        # See cefr_level()'s and this function's own docstring for why.
        # English-only, same reason this whole function's own docstring
        # gives: no Spanish morphy/irregular-exception data exists here.
        if level is None and lang == "en":
            root, root_pos = _morphy_lemma_and_pos(word)
            if root:
                level = cefr_level(root, _WORDNET_POS_NAMES.get(root_pos), strict=True, lang=lang)
        if level is None and lang == "en":
            irregular_root, irregular_pos = _irregular_lemma_and_pos(word)
            if irregular_root:
                level = cefr_level(irregular_root, irregular_pos, strict=True, lang=lang)
        if level is not None:
            cefr_map[word] = level
    return cefr_map


_CEFR_LEVELS_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")


def vocabulary_level_stats(known_words, lang="en"):
    """{level: (known_count, total_count)} across every word in the
    bundled CEFR data for `lang` (cefr_en.json or cefr_es.json), where
    `known_count` is how many of that level's words are in
    `known_words` — the reader's accumulated "known" set for this
    language, from vocabulary_progress.json, built up across every book
    they've run through this app, not just the current one. A word with
    senses at more than one level is counted at its easiest sense here
    (same rule cefr_level() uses with no `pos` given), so each word in
    the data is counted exactly once. Spanish's data has no "C2" entries
    at all (see cefr_data/README.md) — that level's tuple is simply
    (0, 0) for Spanish, same as any level with no data would be.

    English and Spanish only, same restriction as build_cefr_map()
    (returns {} for any other `lang`) since neither data file has any
    other language's vocabulary.

    This is the basis for estimate_vocabulary_level() below; exposed
    separately so a caller that wants the full per-level breakdown (not
    just a single current-level guess) doesn't have to recompute it.
    """
    if lang not in _CEFR_DATA_FILES:
        return {}
    data = _load_cefr_data(lang)
    stats = {level: [0, 0] for level in _CEFR_LEVELS_ORDER}
    for word, entries in data.items():
        level = min(entries.values(), key=lambda lvl: _CEFR_LEVEL_RANK[lvl])
        counts = stats[level]
        counts[1] += 1
        if word in known_words:
            counts[0] += 1
    return {level: tuple(counts) for level, counts in stats.items()}


def estimate_vocabulary_level(known_words, lang="en", threshold=0.5):
    """A rough overall CEFR level for the reader, estimated from how much
    of the bundled data's vocabulary at each level they've marked known
    across every book — NOT a real placement-test result, just a proxy
    built from whatever this app has observed. Returns the highest level
    for which at least `threshold` (default half) of that level's words
    are known, requiring every lower level to also clear the threshold
    first — jumping straight to, say, B2 while barely knowing A2 words
    would be a data artifact (e.g. an A2 word this app happens not to
    have seen the reader read yet), not genuine mastery of A2. Returns
    None for any language but English/Spanish, or if even A1 falls
    short of the threshold (a "beginner"/no-estimate-yet case, not a
    level of its own). Never returns "C2" for Spanish — that level's
    stats are always (0, 0) there (see vocabulary_level_stats()), which
    this loop's own `if total and ...` check already treats as
    unclearable, the same way it would for any language's genuinely
    empty level.
    """
    stats = vocabulary_level_stats(known_words, lang)
    if not stats:
        return None
    current = None
    for level in _CEFR_LEVELS_ORDER:
        known, total = stats[level]
        if total and known / total >= threshold:
            current = level
        else:
            break
    return current


_MORPHY_POS = (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV)


def _morphy_lemma_and_pos(word):
    """(lemma, wn_pos) for the first successful morphy reduction across
    noun/verb/adjective/adverb, tried in that order, or (None, None) if
    `word` is already a base form (or no root could be found at all).
    `wn_pos` is a raw WordNet POS constant (feed it through
    _WORDNET_POS_NAMES for the "noun"/"verb"/... string form) — shared by
    word_family_root() below (which only needs the lemma string, for
    word-family-grouping display) and build_cefr_map()'s stricter
    fallback (which also needs to know WHICH part of speech the
    reduction actually succeeded under — see its docstring for why that
    matters more than a word's own, separately-guessed part of speech).
    """
    for pos in _MORPHY_POS:
        try:
            lemma = wn.morphy(word, pos)
        except LookupError:
            return None, None
        if lemma and lemma != word:
            return lemma, pos
    return None, None


def word_family_root(word, lang="eng"):
    """Return the base/dictionary form of an inflected word — "books",
    "booked", "booking" all resolve to "book" — using WordNet's bundled
    `morphy` analyzer, or None if `word` is already a base form (or no
    root could be found).

    English only: `morphy` is a fixed set of English suffix-stripping
    rules plus an English irregular-forms exception list bundled with
    WordNet itself — it has no equivalent for the other languages this
    app supports (confirmed: it's not part of the multilingual Open
    Multilingual Wordnet data `define()` uses for Spanish/Arabic), so
    callers should only use this when `lang == "eng"`.
    """
    if lang != "eng":
        return None
    lemma, _pos = _morphy_lemma_and_pos(word)
    return lemma


def _should_stop(should_continue):
    return should_continue is not None and not should_continue()


def _tts_cache_dir():
    path = os.path.join(tempfile.gettempdir(), "text_analyzer_tts_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _play_audio_windows(path):
    # ctypes call into winmm.dll's MCI API — no extra playback library
    # needed, confirmed directly to work for MP3 playback. Windows has
    # no standard command-line audio player equivalent to macOS's
    # `afplay`, so this stays a direct API call rather than a subprocess.
    key = os.path.splitext(os.path.basename(path))[0]
    alias = f"tts_{key}"
    winmm = ctypes.windll.winmm
    try:
        rc = winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
        if rc != 0:
            raise RuntimeError(f"MCI open failed (code {rc})")
        rc = winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
        if rc != 0:
            raise RuntimeError(f"MCI play failed (code {rc})")
    finally:
        winmm.mciSendStringW(f"close {alias}", None, 0, None)


def _play_audio_macos(path):
    # afplay ships with every Mac (part of CoreAudio) — no extra
    # dependency needed, same zero-install bar as the Windows path.
    subprocess.run(["afplay", path], check=True)


# No single command-line MP3 player ships with every Linux distro (unlike
# afplay on macOS), so several common ones are tried in turn, quietly,
# until one is found.
_LINUX_AUDIO_PLAYERS = (
    ["mpg123", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpv", "--no-terminal", "--really-quiet"],
    ["cvlc", "--play-and-exit", "--quiet"],
)


def _play_audio_linux(path):
    last_error = None
    for command in _LINUX_AUDIO_PLAYERS:
        try:
            subprocess.run(
                [*command, path], check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
        except FileNotFoundError as exc:
            last_error = exc
        except subprocess.CalledProcessError as exc:
            last_error = exc
    raise RuntimeError(
        "Couldn't play audio: none of mpg123, ffplay, mpv, or vlc were found. "
        "Install one of these to enable pronunciation, e.g. `sudo apt install mpg123`."
    ) from last_error


def _play_audio_android(path):
    # pyjnius binding straight to Android's own MediaPlayer — same
    # gTTS-fetched mp3 as every other platform (no engine swap, full
    # pronunciation/language-coverage parity with Windows/macOS/Linux),
    # just a different playback mechanism, since Android has neither
    # ctypes.windll nor a shell-launchable command-line player. Polls
    # isPlaying() rather than returning immediately after start(), to
    # match every other _play_audio_*'s synchronous-until-done contract
    # (Windows' "play ... wait", macOS/Linux's blocking subprocess.run).
    import time

    from jnius import autoclass

    MediaPlayer = autoclass("android.media.MediaPlayer")
    player = MediaPlayer()
    try:
        player.setDataSource(path)
        player.prepare()
        player.start()
        while player.isPlaying():
            time.sleep(0.1)
    finally:
        player.release()


def speak_word(word, lang):
    """Synthesize and play the pronunciation of a word out loud.

    Uses Google's text-to-speech service (online — the app's one
    accepted online dependency; offline eSpeak-NG was tried first and
    crashed natively for reasons never pinned down, see project history)
    via the `gtts` package to synthesize the MP3, then hands playback off
    to a platform-specific helper: Windows' built-in MCI API (no extra
    library), macOS's bundled `afplay` command, Android's own
    MediaPlayer (via pyjnius), or — Linux having no single standard
    player — the first of a few common command-line players (mpg123/
    ffplay/mpv/vlc) that's actually installed. Verified directly on
    Windows for English, Spanish, Arabic, German, and Turkish; the
    macOS/Linux paths are new and rely on tools standard enough (or,
    for Linux, common enough) that they should carry the same behavior
    over, but haven't been run on those platforms directly.

    Results are cached to a temp directory keyed by (lang, word) so
    replaying the same word — a natural thing to do while studying —
    doesn't re-hit the network every time.

    Raises on failure (e.g. no internet, gTTS doesn't support `lang`, or
    — Linux only — no supported player is installed) rather than failing
    silently, so the caller can show why it didn't work instead of just
    doing nothing.
    """
    key = hashlib.sha1(f"{lang}:{word}".encode("utf-8")).hexdigest()
    path = os.path.join(_tts_cache_dir(), f"{key}.mp3")
    if not os.path.exists(path):
        gTTS(text=word, lang=lang).save(path)

    try:
        from kivy.utils import platform as _kivy_platform
    except ImportError:
        _kivy_platform = None

    if _kivy_platform == "android":
        _play_audio_android(path)
    elif sys.platform == "win32":
        _play_audio_windows(path)
    elif sys.platform == "darwin":
        _play_audio_macos(path)
    else:
        _play_audio_linux(path)


def build_glossary(words, on_progress=None, should_continue=None, wordnet_lang="eng", offline_dict_lang=None):
    """Split words into (word, definition) pairs and words with no match.

    `on_progress(i, total)`, if given, is called after every word instead
    of the default periodic console print — used by vocabulary_app.py to
    drive a progress bar.

    `wordnet_lang` is an NLTK WordNet language code (see
    WORDNET_LANG_CODES) — pass "spa" or "arb" to look words up as Spanish
    or Arabic directly rather than assuming English. Words that aren't in
    WordNet under that language (which, for Arabic especially, is a lot
    of otherwise perfectly ordinary vocabulary — WordNet's Arabic
    coverage is real but noticeably sparse) fall through to
    `offline_dict_lang` if given, then end up undefined.

    Pass `None` (not "eng") for `wordnet_lang` when the book's language
    has no WordNet data at all (e.g. German) — this skips WordNet lookup
    entirely rather than matching against the English dictionary by
    coincidence, which for short common foreign words produces actively
    wrong-looking results (confirmed: German "das" matches an obscure
    English word for a small antelope) that read as "the app is broken"
    rather than merely unhelpful.

    `offline_dict_lang` is an ISO 639-1 code for a bundled offline
    bilingual dictionary (see OFFLINE_DICT_LANGS — currently "de", "ar",
    and "tr") to fall back to for words WordNet didn't match. Unlike
    WordNet's cross-lingual lemma mapping, these are real bilingual
    dictionaries (built from FreeDict for German, from Wiktionary for
    Arabic and Turkish) with definitions written for exactly this
    purpose.

    Words neither WordNet nor the offline dictionary can match are
    simply left undefined — there's no online translate-to-English
    fallback for them. One was built and shipped, then removed: every
    free translation backend tried (Google's scraping backend broke
    outright when Google changed its page structure; PONS broke the same
    way; MyMemory works but is a crowd-sourced sentence-memory database,
    not a dictionary, and gave wrong or garbled results for a real share
    of common words in direct testing — e.g. "çok" as "to be" instead of
    "very", one result even leaking a raw HTML entity into the text)
    fell short of this app's own bar for dictionary quality.

    `should_continue`, if given, is checked before every word, not just
    once up front — confirmed directly that checking only at the start
    (an earlier version of this function) let a Stop/Cancel click during
    a run set the flag but never actually interrupt the loop, so the
    whole word list kept getting looked up regardless. Returning False
    stops before the next word is processed; every word not yet reached
    is added to `undefined` (it was never looked up, not "not found",
    but this keeps existing defined/undefined counts meaningful for a
    stopped run without a third bucket).
    """
    if _should_stop(should_continue):
        return [], list(words), None
    defined = []
    undefined = []
    total = len(words)

    offline_conn = None
    offline_path = _offline_dict_path(offline_dict_lang) if offline_dict_lang else None
    if offline_path:
        offline_conn = sqlite3.connect(offline_path)
    offline_first = offline_dict_lang in OFFLINE_DICT_PRIORITY_LANGS

    def _offline_lookup(word):
        row = offline_conn.execute(
            "SELECT definition FROM dict WHERE word = ?", (word,)
        ).fetchone()
        return row[0] if row else None

    try:
        for i, word in enumerate(words, start=1):
            if _should_stop(should_continue):
                # Words not yet reached weren't looked up, not "not
                # found" — but folding them into undefined here matches
                # this function's own early-return-before-the-loop
                # behavior above, and needs no change to how callers
                # already read/display defined vs undefined counts for a
                # mid-run Stop.
                undefined.extend(words[i - 1:])
                break
            if on_progress:
                on_progress(i, total)
            elif i % 500 == 0:
                print(f"  looked up {i}/{total} words...")
            definition = None
            if offline_first and offline_conn:
                definition = _offline_lookup(word)
            if not definition and wordnet_lang:
                definition = define(word, lang=wordnet_lang)
            if not definition and not offline_first and offline_conn:
                definition = _offline_lookup(word)
            if definition:
                defined.append((word, definition))
            else:
                undefined.append(word)
    finally:
        if offline_conn:
            offline_conn.close()

    return defined, undefined, None


def main():
    filepath = input("Path to your .pdf or .epub file: ").strip()

    print("Reading book...")
    try:
        text = load_book_text(filepath)
    except NoExtractableTextError as exc:
        # This is the one error load_book_text() raises deliberately,
        # with a message already written for a reader to act on (try OCR,
        # or a different copy of the file) — the GUI (vocabmaster/app.py)
        # already catches this, but main() here (this file's standalone
        # CLI entry point) didn't, so it surfaced as a raw traceback
        # instead of that message.
        print(f"\n{exc}")
        return
    except OSError as exc:
        print(f"\nCouldn't read '{filepath}': {exc}")
        return

    print("Extracting unique words...")
    words = extract_unique_words(text)
    print(f"Found {len(words)} unique words.")

    lang = detect_language(text)
    if lang != "en":
        print(f"Detected language: {language_name(lang)}.")

    ensure_wordnet()

    print("Looking up definitions...")
    defined, undefined, _ = build_glossary(words)
    if lang != "en" and undefined:
        print(
            f"{len(undefined)} words had no dictionary match (likely {language_name(lang)} words "
            "WordNet doesn't cover)."
        )

    report_lines = []
    report_lines.append("VOCABULARY GLOSSARY")
    report_lines.append(f"Source file: {filepath}")
    report_lines.append(f"Unique words found: {len(words)}")
    report_lines.append(f"Words with a definition: {len(defined)}")
    report_lines.append(f"Words with no dictionary match: {len(undefined)}")
    report_lines.append("")
    report_lines.append("DEFINITIONS")
    for word, definition in defined:
        report_lines.append(f"  {word}: {definition}")

    if undefined:
        report_lines.append("")
        report_lines.append("WORDS WITH NO DICTIONARY MATCH (names, jargon, typos, etc.)")
        for word in undefined:
            report_lines.append(f"  {word}")

    report = "\n".join(report_lines)

    output_path = "vocabulary_glossary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved glossary of {len(defined)} defined words to: {output_path}")


if __name__ == "__main__":
    main()
