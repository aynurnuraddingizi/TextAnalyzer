"""Word Info window: pronunciation (IPA), lexical relations (synonyms/
antonyms/hypernyms/hyponyms/meronyms/holonyms via WordNet), and
collocations (words that occur near this one elsewhere in the SAME
book) for a single word — the deeper linguistic detail the compact
Detail panel has no room for, computed on demand rather than for every
word up front (nothing here is needed to render the main word table,
unlike part-of-speech/CEFR level, so there's no pipeline cost to a
reader who never opens this window).
"""
import tkinter as tk
from tkinter import ttk

from text_analyzer import ensure_linguistics_data, find_collocations, word_pronunciation, word_relations

import theme as th

# (data key, display label) — in the order shown; a relation with no
# entries for this word is skipped entirely rather than printed empty,
# so a word with only, say, hypernyms doesn't show five "(none)" lines.
_RELATION_LABELS = (
    ("synonyms", "Synonyms"),
    ("antonyms", "Antonyms"),
    ("hypernyms", "Broader term"),
    ("hyponyms", "More specific"),
    ("meronyms", "Made of / part of"),
    ("holonyms", "Part of a..."),
)
# How many words to list per relation before truncating with "…" — a
# common hypernym chain (e.g. "publication") can have dozens of
# hyponyms; this keeps the window scannable rather than dumping all of
# them, same "cap, don't drop silently" spirit as this app's other
# capped lists (concordance's MAX_RESULTS, a grammar rule's sentence cap).
_MAX_RELATION_ITEMS = 12


class WordInfoWindow(tk.Toplevel):
    """`lang` is the book's 2-letter ISO code (e.g. "en") — gates
    pronunciation, still genuinely English-only (the CMU Pronouncing
    Dictionary has no other-language data; see word_pronunciation()'s
    docstring). Collocations are NOT gated on `lang` beyond what
    find_collocations() itself already handles internally: it's a plain
    statistical count over the book's own sentences that works for any
    language, and degrades gracefully (just a noisier result, not an
    error) for a language with no stopword list to filter with — passed
    through as-is rather than special-cased here. `wordnet_lang` is the
    3-letter WordNet code (e.g. "eng"/"spa"/"arb"/"fra", from
    WORDNET_LANG_CODES) — gates word relations, which work for any
    WordNet-backed language, not just English. `sentences` is the book's
    cached split_sentences() list (self.book_sentences in app.py) —
    reused, not re-split, same as every other feature that needs the
    book's sentences.
    """

    def __init__(self, master, word, palette, lang="en", wordnet_lang=None, sentences=None):
        super().__init__(master)
        self.p = palette
        self.title(f"Word Info — {word}")
        self.geometry("520x580")
        self.minsize(420, 400)
        self.configure(bg=self.p["BG"])

        tk.Label(self, text=word, bg=self.p["BG"], fg=self.p["ACCENT"], font=th.FONT_TITLE).pack(
            anchor="w", padx=20, pady=(16, 0),
        )

        body_frame = tk.Frame(
            self, bg=self.p["PANEL"], highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"],
        )
        body_frame.pack(fill="both", expand=True, padx=20, pady=(10, 14))
        scrollbar = tk.Scrollbar(
            body_frame, bg=self.p["PANEL"], troughcolor=self.p["BG"], activebackground=self.p["SECOND_HOVER"],
            highlightthickness=0,
        )
        scrollbar.pack(side="right", fill="y")
        text = tk.Text(
            body_frame, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word", relief="flat",
            yscrollcommand=scrollbar.set, highlightthickness=0, padx=14, pady=12, cursor="xterm",
            insertbackground=self.p["FG"],
        )
        text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text.yview)
        text.tag_configure("heading", font=th.FONT_BOLD, foreground=self.p["FG"])
        text.tag_configure("label", font=th.FONT_SMALL_BOLD, foreground=self.p["DIM"])
        text.tag_configure("value", font=th.FONT, foreground=self.p["FG"])
        text.tag_configure("dim", font=th.FONT_SMALL_ITALIC, foreground=self.p["DIM"])
        text.tag_configure("ipa", font=th.FONT_CARD_WORD, foreground=self.p["ACCENT"])

        # Unconditional, not "if lang == 'en'": ensure_linguistics_data()
        # also covers the stopwords corpus collocations needs for every
        # language now, not just the CMU dict English pronunciation
        # alone still needs — see its own docstring.
        ensure_linguistics_data()

        text.insert("end", "PRONUNCIATION\n", "heading")
        if lang in ("en", "es"):
            pron = word_pronunciation(word, lang)
            if pron:
                text.insert("end", f"/{pron['ipa']}/", "ipa")
                plural = "" if pron["syllables"] == 1 else "s"
                text.insert("end", f"   ({pron['syllables']} syllable{plural})\n", "dim")
                if lang == "es":
                    text.insert(
                        "end",
                        "Latin American Spanish (seseo, yeísmo) — rule-based, not looked up.\n\n",
                        "dim",
                    )
                else:
                    text.insert("end", "\n", "dim")
            elif lang == "en":
                text.insert(
                    "end",
                    "Not in the CMU Pronouncing Dictionary — likely a proper noun, a rare/technical "
                    "word, or an inflected form it doesn't list separately.\n\n",
                    "dim",
                )
            else:
                text.insert("end", "Couldn't transcribe this word.\n\n", "dim")
        else:
            text.insert("end", "Pronunciation data is only available for English and Spanish.\n\n", "dim")

        text.insert("end", "WORD RELATIONS\n", "heading")
        relations = word_relations(word, wordnet_lang) if wordnet_lang else None
        if relations is None:
            text.insert("end", "No WordNet data for this word.\n\n", "dim")
        else:
            shown_any = False
            for key, label in _RELATION_LABELS:
                values = relations.get(key) or []
                if not values:
                    continue
                shown_any = True
                shown = values[:_MAX_RELATION_ITEMS]
                suffix = f", + {len(values) - _MAX_RELATION_ITEMS} more" if len(values) > _MAX_RELATION_ITEMS else ""
                text.insert("end", f"{label}: ", "label")
                text.insert("end", ", ".join(shown) + suffix + "\n", "value")
            if not shown_any:
                text.insert("end", "WordNet has an entry for this word but lists no relations of any kind for it.\n", "dim")
            text.insert("end", "\n")

        text.insert("end", "COMMON NEARBY WORDS IN THIS BOOK\n", "heading")
        if not sentences:
            text.insert("end", "No book text available.\n", "dim")
        else:
            collocations = find_collocations(sentences, word, lang=lang)
            if collocations:
                for neighbor, count in collocations:
                    text.insert("end", f"  {neighbor}", "value")
                    text.insert("end", f"   ({count}×)\n", "dim")
            else:
                text.insert("end", "Not enough occurrences in this book to find a pattern.\n", "dim")

        text.config(state="disabled")

        close_row = tk.Frame(self, bg=self.p["BG"])
        close_row.pack(fill="x", padx=20, pady=(0, 16))
        ttk.Button(close_row, text="CLOSE", command=self.destroy).pack(side="right")

        self.bind("<Escape>", lambda _e: self.destroy())
        th.apply_dark_titlebar(self, self.p is th.DARK)
