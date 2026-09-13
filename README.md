# Text Analyzer

A Windows desktop app that reads a PDF or EPUB — in any language — and
builds you a vocabulary glossary, a grammar breakdown, and a phrase list
from it, with offline dictionary lookups, CEFR levels, spaced-repetition
flashcards, and audio pronunciation.

## What it does

- Extracts every distinct word from a `.pdf` or `.epub` and looks up a
  definition for each one — offline, no internet required after setup.
  English uses NLTK's WordNet; German, Arabic, Turkish, Spanish, and
  Russian use bundled offline dictionaries (see `dict_data/README.md`).
- Detects the book's language automatically.
- Tags each word with part of speech and CEFR level (A1–C2) where known.
- Finds grammar structures (verb tenses, clause types, etc.) and phrases/
  idioms, each linked back to the sentences they occur in.
- Tracks which words/rules you already know vs. are still learning, with
  SM-2 spaced-repetition scheduling.
- Flashcard and writing-quiz study modes, launchable for your whole
  glossary or for just a handful of words you've selected ("Focus View").
- Text-to-speech pronunciation for any word.
- Export to CSV, JSON, TXT, DOCX, PDF, Markdown, or an Anki-importable
  deck.

## Download and run (no Python required)

Grab the latest `TextAnalyzer.exe` from this repo's
[Releases](../../releases) page and run it — nothing else to install.

## Run from source

Requires Python 3.10+.

```
pip install -r requirements.txt
python vocabmaster/main.py
```

The first run downloads NLTK's WordNet data automatically (needs internet
once; cached locally after that). Everything else — including all lookups
during normal use — works fully offline.

Pronunciation audio (`gTTS`) needs an internet connection at the moment
you press the speaker button; nothing else does.

### Building the .exe yourself

```
pip install pyinstaller
pyinstaller TextAnalyzer.spec --noconfirm
```

The built executable is written to `dist/TextAnalyzer.exe`.

## Project layout

- `text_analyzer.py` — core engine: text extraction, dictionary lookups,
  pronunciation, grammar/phrase detection, progress persistence.
- `vocabmaster/` — the Tkinter application (UI, study modes, export).
- `dict_data/`, `cefr_data/`, `phrase_data/` — scripts and sources used to
  build the bundled offline dictionaries/CEFR lists/phrase data, each
  with its own `README.md` explaining where that data came from.

## Platform notes

Built and tested on Windows. The core engine (`text_analyzer.py`) has no
Windows-specific code and runs on macOS/Linux too; the packaged `.exe`
and DPI-awareness handling in `vocabmaster/main.py` are Windows-only.
On Linux, pronunciation playback needs one of `mpg123`, `ffplay`, `mpv`,
or `vlc` installed (e.g. `sudo apt install mpg123`).

## License

MIT — see [LICENSE](LICENSE).
