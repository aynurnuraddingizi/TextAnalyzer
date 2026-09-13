"""Main window: topbar, recent-books sidebar, word table + detail panel,
docked study/stats column, status bar. The extraction/glossary pipeline,
threading, and queue-polling logic here is carried over from the
original vocabulary_app.py near-verbatim — this file changes the widget
layer, not the backend calls or the app's actual behavior.

No translate-to-English feature: tried and removed. Every free backend
available (Google's scraping backend, which broke outright when Google
changed its page structure; PONS, also broken the same way; MyMemory,
which works but is a crowd-sourced sentence-memory database rather than
a dictionary and gave wrong or garbled results for a meaningful share of
real words in direct testing) fell short of this app's own bar for
offline dictionary quality — confirmed directly, not assumed.
"""
import ctypes
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from text_analyzer import (
    OFFLINE_DICT_LANGS,
    RESET_BACKUP_DAYS,
    WORDNET_LANG_CODES,
    add_recent_book,
    app_dir,
    backup_before_reset,
    build_example_sentences,
    build_cefr_map,
    build_glossary,
    build_pos_map,
    cefr_level,
    define,
    detect_language,
    due_words,
    empty_language_progress,
    ensure_phrase_data,
    ensure_wordnet,
    estimate_vocabulary_level,
    extract_phrases,
    find_occurrences,
    language_name,
    load_book_text,
    load_progress,
    load_settings,
    mark_known,
    readability_stats,
    remove_recent_book,
    restorable_backup,
    review_word,
    save_progress,
    save_settings,
    speak_word,
    split_sentences,
    word_family_root,
    word_frequencies,
    word_pos,
    words_learned_since,
)

import theme as th
import export as exp
from study import StudyWindow, build_study_summary_card
from onboarding import OnboardingWizard
from concordance import ConcordanceWindow
from word_info import WordInfoWindow
from phrase_analyzer import build_phrase_section, populate_phrase_tree
from focus_view import FocusViewWindow
from grammar_analyzer import (
    CATEGORY_ORDER,
    CATEGORY_ORDER_ES,
    GrammarQuizWindow,
    analyze_grammar,
    build_grammar_readiness_card,
    build_grammar_section,
    build_sentence_index,
    ensure_pos_tagger,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(os.path.dirname(BASE_DIR), "text_analyzer_icon.ico")

SELECTABLE_LANGUAGES = [
    ("auto", "Auto-detect", None),
    ("en", "English", "en"),
    ("es", "Spanish", "es"),
    ("fr", "French", "fr"),
    ("ar", "Arabic (Qur'an language)", "ar"),
    ("de", "German", "de"),
    ("tr", "Turkish", "tr"),
    ("ru", "Russian", "ru"),
]

STATUS_ICON = {"known": "✅", "learning": "\U0001F4D6", "new": ""}


class MainWindow(tk.Tk):
    def _apply_dpi_scaling(self):
        # main.py already declares this process per-monitor DPI aware
        # (before any Tk window exists) so Windows stops bitmap-stretching
        # the whole window on a scaled display. That alone leaves Tk
        # still assuming 96 DPI internally, so point-sized fonts render
        # too small relative to the display's real pixel density — this
        # tells Tk the real DPI so fonts and other point-based sizing
        # come out correctly proportioned instead of merely crisp-but-tiny.
        if sys.platform != "win32":
            return
        try:
            dpi = ctypes.windll.user32.GetDpiForSystem()
            self.tk.call("tk", "scaling", dpi / 72)
        except Exception:
            pass

    def __init__(self):
        super().__init__()
        self._apply_dpi_scaling()
        self.title("Text Analyzer")
        self.geometry("1360x820")
        self.minsize(1100, 680)
        if os.path.exists(ICON_PATH):
            try:
                self.iconbitmap(ICON_PATH)
            except tk.TclError:
                pass

        self.settings = load_settings()
        self.palette = th.PALETTES.get(self.settings.get("theme", "light"), th.LIGHT)
        self.themed = th.ThemedWidgetRegistry()

        self.filepath = None
        self.book_text = None
        self.detected_lang = None
        self.glossary_entries = []
        self.undefined_words = []
        self.word_freqs = {}
        self.word_examples = {}
        self.word_pos_map = {}     # word -> "noun"/"verb"/etc., only for WordNet-backed languages
        self.word_cefr_map = {}    # word -> "A1".."C2", English/Spanish only — see build_cefr_map()'s docstring
        self.word_lookup = {}      # word -> (definition, count, example, pos, cefr) for the detail panel
        self.family_map = {}            # root -> sorted [word, ...] (only roots with 2+ members)
        self.word_family_root_of = {}   # word -> root, only for words in a multi-member family
        self.family_definition = {}     # root -> definition or None
        self.family_pos = {}            # root -> "noun"/"verb"/etc. or None
        self.family_cefr = {}           # root -> "A1".."C2" or None
        self.book_sentences = []        # cached split_sentences(text), for the Find in Book panel
        # rule_id -> {name, category, definition, rule_explanation, sentences, total_count} — also directly
        # usable as the Grammar tree's per-rule lookup, since a rule row's tree iid IS its rule_id.
        self.grammar_results = {}
        self.selected_grammar_rule = None
        # {length: {"entries": [{"phrase","count","pmi","example"}, ...], "total_found": int}}
        # — extract_phrases()'s own return shape, kept as-is (app.py never restructures it).
        self.phrases = {}
        self.selected_phrase = None   # (length, phrase_text) or None
        self.active_section = "vocab"   # "vocab"/"grammar"/"phrases" — which center-slot content is showing
        self.sort_by_freq = True
        self.selected_word = None   # the single selected word, or None if 0 or 2+ are selected
        self.selected_words = []    # every currently-selected word (1 or many)
        self.lang_choice_var = tk.StringVar(value="auto")
        self.hide_known_var = tk.BooleanVar(value=True)
        self.group_families_var = tk.BooleanVar(value=False)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self.render_results())
        # English/Spanish only (CEFR level itself is — see
        # build_cefr_map()'s docstring), same "disabled, not hidden"
        # treatment as group_families_check for any other book language.
        self.level_filter_var = tk.StringVar(value="All levels")
        self.phrase_search_var = tk.StringVar()
        self.phrase_search_var.trace_add("write", lambda *_: self._render_phrases())

        progress = load_progress()
        # Every language's progress, all at once — {"en": {"known": set,
        # "learning": set, "schedule": dict, "learned_at": dict}, "de":
        # {...}, ...}. self.known_words/learning_words/progress_schedule/
        # progress_learned_at below are never their own independent data
        # — they're live references into whichever language bucket is
        # currently active (see _switch_active_language()), so every
        # existing bit of code that mutates them in place (.add(),
        # .discard(), |=, review_word(), mark_known()) keeps working
        # completely unchanged; only a REASSIGNMENT of one of those four
        # names (rather than mutating it) would silently break the link.
        self.all_progress = progress["languages"]
        self.reset_backups = progress["reset_backups"]  # lang -> pre-reset snapshot, see reset_progress()
        self.active_lang = None
        self.known_words = set()
        self.learning_words = set()
        self.progress_schedule = {}
        self.progress_learned_at = {}
        self._switch_active_language("en")  # default before any book's language is known

        # Grammar's own known/learning/schedule/learned_at — a SINGLE
        # global bucket, not per-language like the vocabulary progress
        # above, since grammar analysis is English-only (see
        # grammar_analyzer.analyze_grammar()'s docstring): there's never
        # a second language's grammar progress to keep separate from a
        # first one. Loaded once here, not per-book, since knowing
        # "Present Perfect" isn't scoped to any one book. Reset/restore
        # reuse backup_before_reset()/restorable_backup() against
        # self.reset_backups under the sentinel key "grammar" (never a
        # real ISO 639-1 language code, so it can't collide with one).
        grammar_progress = progress["grammar"]
        self.grammar_known = grammar_progress["known"]
        self.grammar_learning = grammar_progress["learning"]
        self.grammar_schedule = grammar_progress["schedule"]
        self.grammar_learned_at = grammar_progress["learned_at"]

        self.work_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.run_start_time = None

        style = ttk.Style(self)
        th.build_ttk_style(style, self.palette)
        self._build_widgets()
        self._refresh_recent_books()
        self._refresh_vocab_stat()
        self._refresh_reading_readiness()
        self._refresh_progress_stat()
        self._refresh_grammar_stat()
        self._refresh_grammar_readiness()
        self._refresh_grammar_progress_stat()
        th.apply_dark_titlebar(self, self.settings.get("theme") == "dark")
        self.after(100, self.poll_queue)
        self.after(500, self._tick_elapsed)
        self.after(150, self._repaint_disabled_buttons)
        if not self.settings.get("onboarded"):
            self.after(300, self.show_onboarding)

    def _repaint_disabled_buttons(self):
        # Confirmed directly: on first launch, STOP (created disabled,
        # never yet toggled) sometimes paints as a stray unthemed white
        # box instead of its correct dark styling — an isolated repro
        # with the same style/state setup never reproduced it, so this
        # looks like a one-time stale-paint race during this window's
        # heavier startup (dictionaries, NLTK, DWM title bar call) rather
        # than a real style bug. Bouncing state off and back on forces
        # ttk to recompute and repaint, which reliably clears it.
        for btn in (
            self.stop_btn, self.run_btn, self.save_btn, self.anki_btn, self.concordance_btn,
            self.study_flashcard_btn, self.study_writing_btn, self.study_anki_btn, self.restore_btn,
            self.detail_speak_btn, self.detail_find_btn, self.detail_word_info_btn, self.detail_focus_btn,
            self.detail_know_btn, self.detail_learning_btn,
        ):
            was_disabled = "disabled" in btn.state()
            if was_disabled:
                btn.state(["!disabled"])
                btn.state(["disabled"])

    # ------------------------------------------------------------- layout

    def _build_widgets(self):
        p = self.palette
        self.configure(bg=p["BG"])

        self._build_topbar()
        self._build_section_switcher()

        middle = tk.Frame(self, bg=p["BG"])
        middle.pack(fill="both", expand=True)
        self.themed.add(middle, bg="BG")

        self._build_sidebar(middle)     # packed side=left first (fixed width)
        self._build_right_column(middle)  # packed side=right (fixed width)
        self._build_center(middle)      # fills remaining space

        self._build_statusbar()

    def _build_topbar(self):
        p = self.palette
        bar = tk.Frame(self, bg=p["BG"])
        bar.pack(fill="x", padx=20, pady=(16, 4))
        self.themed.add(bar, bg="BG")

        title_box = tk.Frame(bar, bg=p["BG"])
        title_box.pack(side="left")
        self.themed.add(title_box, bg="BG")
        title_lbl = tk.Label(title_box, text="TEXT ANALYZER", bg=p["BG"], fg=p["FG"], font=th.FONT_TITLE)
        title_lbl.pack(anchor="w")
        self.themed.add(title_lbl, bg="BG", fg="FG")
        subtitle_lbl = tk.Label(
            title_box,
            text="Extract every word from a PDF or EPUB and look up its definition — instant, offline.",
            bg=p["BG"], fg=p["DIM"], font=th.FONT,
        )
        subtitle_lbl.pack(anchor="w")
        self.themed.add(subtitle_lbl, bg="BG", fg="DIM")

        icons = tk.Frame(bar, bg=p["BG"])
        icons.pack(side="right")
        self.themed.add(icons, bg="BG")
        self.theme_btn = ttk.Button(icons, text=self._theme_icon(), style="Icon.TButton", command=self.toggle_theme)
        self.theme_btn.pack(side="right", padx=(6, 0))
        ttk.Button(icons, text="❓", style="Icon.TButton", command=self._show_help).pack(side="right", padx=(6, 0))
        ttk.Button(icons, text="🎓", style="Icon.TButton", command=self.show_onboarding).pack(side="right", padx=(6, 0))
        ttk.Button(icons, text="⚙", style="Icon.TButton", command=self._show_settings).pack(side="right", padx=(6, 0))

    def _theme_icon(self):
        return "☀" if self.palette is th.DARK else "\U0001F319"

    # ---- section switcher: Vocabulary vs Grammar ------------------------

    def _build_section_switcher(self):
        # Hand-built ttk.Buttons, not a ttk.Notebook: nothing else in
        # this app uses a stock ttk compound widget for primary
        # navigation, and Notebook's native tab chrome doesn't take this
        # app's palette as cleanly as a plain themed button does.
        p = self.palette
        bar = tk.Frame(self, bg=p["BG"])
        bar.pack(fill="x", padx=20, pady=(0, 4))
        self.themed.add(bar, bg="BG")
        self.vocab_section_btn = ttk.Button(
            bar, text="\U0001F4DA VOCABULARY", command=lambda: self._show_section("vocab"),
        )
        self.vocab_section_btn.pack(side="left")
        self.grammar_section_btn = ttk.Button(
            bar, text="\U0001F524 GRAMMAR", command=lambda: self._show_section("grammar"),
        )
        self.grammar_section_btn.pack(side="left", padx=(8, 0))
        self.phrase_section_btn = ttk.Button(
            bar, text="\U0001F4DD PHRASES", command=lambda: self._show_section("phrases"),
        )
        self.phrase_section_btn.pack(side="left", padx=(8, 0))
        self._update_section_buttons()

    def _update_section_buttons(self):
        self.vocab_section_btn.config(style="Accent.TButton" if self.active_section == "vocab" else "TButton")
        self.grammar_section_btn.config(style="Accent.TButton" if self.active_section == "grammar" else "TButton")
        self.phrase_section_btn.config(style="Accent.TButton" if self.active_section == "phrases" else "TButton")

    def _show_section(self, name):
        self.active_section = name
        # Phrases has no progress-tracking/quiz concept of its own (see
        # build_phrase_section()'s docstring for why that's a deliberate
        # v1 scope cut, not an oversight), so it has no dedicated right-
        # column/sidebar cards to raise — it reuses the vocab ones,
        # which show real, still-relevant info (this book's vocabulary
        # stats) rather than something misleadingly grammar- or
        # phrase-specific.
        center = {"vocab": self.vocab_center_frame, "grammar": self.grammar_center_frame,
                  "phrases": self.phrase_center_frame}[name]
        side = {"vocab": self.vocab_right_frame, "grammar": self.grammar_right_frame,
                "phrases": self.vocab_right_frame}[name]
        stats = {"vocab": self.vocab_stats_frame, "grammar": self.grammar_stats_frame,
                 "phrases": self.vocab_stats_frame}[name]
        center.tkraise()
        side.tkraise()
        stats.tkraise()
        self._update_section_buttons()
        self._refresh_export_buttons()

    # ---- sidebar (left): recent books + global stats -------------------

    def _build_sidebar(self, parent):
        p = self.palette
        sidebar = tk.Frame(parent, bg=p["BG"], width=230)
        sidebar.pack(side="left", fill="y", padx=(20, 10), pady=10)
        sidebar.pack_propagate(False)
        self.themed.add(sidebar, bg="BG")

        header = tk.Label(sidebar, text="RECENT BOOKS", bg=p["BG"], fg=p["FG"], font=th.FONT_BOLD)
        header.pack(anchor="w")
        self.themed.add(header, bg="BG", fg="FG")

        self.recent_list_frame = tk.Frame(sidebar, bg=p["BG"])
        self.recent_list_frame.pack(fill="both", expand=True, pady=(6, 10))
        self.themed.add(self.recent_list_frame, bg="BG")

        ttk.Button(sidebar, text="+ ADD BOOK", command=self.select_pdf).pack(fill="x")

        # Below "+ ADD BOOK", the stat cards + reset/restore buttons
        # swap between vocabulary and grammar content depending on
        # self.active_section — same place()+tkraise() pattern
        # _build_center() already uses for the main content area,
        # applied here too so switching sections actually changes the
        # sidebar, not just the center panel. "RECENT BOOKS"/"+ ADD
        # BOOK" above stay shared: you need a book loaded either way.
        stats_container = tk.Frame(sidebar, bg=p["BG"])
        stats_container.pack(fill="both", expand=True, pady=(16, 0))
        self.themed.add(stats_container, bg="BG")

        self.vocab_stats_frame = tk.Frame(stats_container, bg=p["BG"])
        self.vocab_stats_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.themed.add(self.vocab_stats_frame, bg="BG")
        self._build_vocab_stats(self.vocab_stats_frame)

        self.grammar_stats_frame = tk.Frame(stats_container, bg=p["BG"])
        self.grammar_stats_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.themed.add(self.grammar_stats_frame, bg="BG")
        self._build_grammar_stats(self.grammar_stats_frame)

        self.vocab_stats_frame.tkraise()

    def _build_vocab_stats(self, sidebar):
        p = self.palette
        stats_frame = tk.Frame(
            sidebar, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
        )
        stats_frame.pack(fill="x")
        self.themed.add(stats_frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        # Text set per-language in _refresh_vocab_stat() (e.g. "YOUR
        # VOCABULARY — ENGLISH") — placeholder here, never actually seen.
        self.vocab_header_label = tk.Label(
            stats_frame, text="YOUR VOCABULARY", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD,
        )
        self.vocab_header_label.pack(anchor="w", padx=10, pady=(10, 4))
        self.themed.add(self.vocab_header_label, bg="PANEL", fg="FG")
        self.vocab_stat_label = tk.Label(
            stats_frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT, wraplength=190, justify="left",
        )
        self.vocab_stat_label.pack(anchor="w", padx=10, pady=(0, 10))
        self.themed.add(self.vocab_stat_label, bg="PANEL", fg="DIM")

        progress_frame = tk.Frame(
            sidebar, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
        )
        progress_frame.pack(fill="x", pady=(16, 0))
        self.themed.add(progress_frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        # Text set per-language in _refresh_progress_stat() — placeholder.
        self.progress_header_label = tk.Label(
            progress_frame, text="PROGRESS", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD,
        )
        self.progress_header_label.pack(anchor="w", padx=10, pady=(10, 4))
        self.themed.add(self.progress_header_label, bg="PANEL", fg="FG")
        self.progress_stat_label = tk.Label(
            progress_frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT, wraplength=190, justify="left",
        )
        self.progress_stat_label.pack(anchor="w", padx=10, pady=(0, 10))
        self.themed.add(self.progress_stat_label, bg="PANEL", fg="DIM")

        # Text set per-language in _refresh_vocab_stat() (e.g. "RESET
        # ENGLISH PROGRESS") — resets only the currently active
        # language's known/learning history, never every language at
        # once, now that they're tracked separately.
        self.reset_btn = ttk.Button(sidebar, text="RESET PROGRESS", command=self.reset_progress)
        self.reset_btn.pack(fill="x", pady=(10, 0))
        # Enabled only when a same-language backup from a recent reset
        # exists (see _refresh_restore_button()) — undoes exactly that
        # last reset, within RESET_BACKUP_DAYS of it happening.
        self.restore_btn = ttk.Button(sidebar, text="RESTORE PROGRESS", command=self.restore_progress, state="disabled")
        self.restore_btn.pack(fill="x", pady=(6, 0))

    def _build_grammar_stats(self, sidebar):
        p = self.palette
        stats_frame = tk.Frame(
            sidebar, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
        )
        stats_frame.pack(fill="x")
        self.themed.add(stats_frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        self.grammar_header_label = tk.Label(
            stats_frame, text="YOUR GRAMMAR", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD,
        )
        self.grammar_header_label.pack(anchor="w", padx=10, pady=(10, 4))
        self.themed.add(self.grammar_header_label, bg="PANEL", fg="FG")
        self.grammar_stat_label = tk.Label(
            stats_frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT, wraplength=190, justify="left",
        )
        self.grammar_stat_label.pack(anchor="w", padx=10, pady=(0, 10))
        self.themed.add(self.grammar_stat_label, bg="PANEL", fg="DIM")

        grammar_progress_frame = tk.Frame(
            sidebar, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
        )
        grammar_progress_frame.pack(fill="x", pady=(16, 0))
        self.themed.add(grammar_progress_frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        self.grammar_progress_header_label = tk.Label(
            grammar_progress_frame, text="GRAMMAR PROGRESS", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD,
        )
        self.grammar_progress_header_label.pack(anchor="w", padx=10, pady=(10, 4))
        self.themed.add(self.grammar_progress_header_label, bg="PANEL", fg="FG")
        self.grammar_progress_stat_label = tk.Label(
            grammar_progress_frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT, wraplength=190, justify="left",
        )
        self.grammar_progress_stat_label.pack(anchor="w", padx=10, pady=(0, 10))
        self.themed.add(self.grammar_progress_stat_label, bg="PANEL", fg="DIM")

        # "RESET GRAMMAR"/"RESTORE GRAMMAR", not the longer "...
        # PROGRESS" — same precedent _refresh_vocab_stat() already
        # documents for its own RESET button (extra words overflowed
        # the sidebar's fixed width and got cut off), applied here too.
        self.grammar_reset_btn = ttk.Button(
            sidebar, text="RESET GRAMMAR", command=self.reset_grammar_progress,
        )
        self.grammar_reset_btn.pack(fill="x", pady=(10, 0))
        self.grammar_restore_btn = ttk.Button(
            sidebar, text="RESTORE GRAMMAR", command=self.restore_grammar_progress, state="disabled",
        )
        self.grammar_restore_btn.pack(fill="x", pady=(6, 0))

    def _refresh_recent_books(self):
        for child in self.recent_list_frame.winfo_children():
            child.destroy()
        p = self.palette
        recent = self.settings.get("recent_books", [])
        if not recent:
            lbl = tk.Label(
                self.recent_list_frame, text="No books yet.", bg=p["BG"], fg=p["DIM"], font=th.FONT,
            )
            lbl.pack(anchor="w")
            self.themed.add(lbl, bg="BG", fg="DIM")
            return
        for book in recent:
            path = book.get("path")
            title = book.get("title", "?")
            if len(title) > 22:
                title = title[:19] + "..."
            row = tk.Frame(self.recent_list_frame, bg=p["BG"])
            row.pack(fill="x", pady=2)
            self.themed.add(row, bg="BG")
            # Packed before the title label (which expands to fill the
            # rest of the row) so the remove button keeps its own fixed
            # width instead of being squeezed out — same rule used
            # everywhere else in this file for a fixed-width sibling
            # next to an expanding one.
            remove_btn = tk.Label(
                row, text="✕", bg=p["BG"], fg=p["DIM"], font=th.FONT, cursor="hand2", padx=4,
            )
            remove_btn.pack(side="right")
            remove_btn.bind("<Button-1>", lambda _e, path=path: self._remove_recent_book(path))
            self.themed.add(remove_btn, bg="BG", fg="DIM")

            title_lbl = tk.Label(
                row, text=f"\U0001F4D5 {title}", bg=p["BG"], fg=p["FG"], font=th.FONT,
                anchor="w", cursor="hand2", wraplength=170, justify="left",
            )
            title_lbl.pack(side="left", fill="x", expand=True)
            title_lbl.bind("<Button-1>", lambda _e, path=path: self._open_recent_book(path))
            self.themed.add(title_lbl, bg="BG", fg="FG")

    def _open_recent_book(self, path):
        if not path or not os.path.exists(path):
            messagebox.showinfo("Recent book", "That file couldn't be found — it may have moved or been deleted.")
            return
        self._select_book_path(path)

    def _remove_recent_book(self, path):
        # No confirmation dialog: this only forgets the shortcut, it
        # doesn't touch the actual file — reopening the book (SELECT
        # BOOK) puts it right back, same low-stakes/reversible bar as
        # the "Hide words I already know" checkbox elsewhere.
        self.settings = remove_recent_book(self.settings, path)
        save_settings(self.settings)
        self._refresh_recent_books()

    # ---- right column: docked study summary + stats/chart --------------

    def _build_right_column(self, parent):
        # Same swappable-frame pattern as _build_center()/the sidebar's
        # stats: the docked card + chart change with the active section
        # instead of staying permanently vocabulary-only.
        p = self.palette
        # 340, not 300: confirmed directly (screenshot) that at 300 the
        # grammar card's longer button labels ("IDENTIFY IN SENTENCE",
        # "EXPORT GRAMMAR TO ANKI") rendered wider than their own button
        # and got cut off mid-word — ttk.Button doesn't wrap or shrink
        # text to fit, so the fix is real headroom, not a label tweak
        # alone (their text was shortened too, but this app's font/DPI
        # still needs the extra room to be safe against a scaled display).
        col = tk.Frame(parent, bg=p["BG"], width=340)
        col.pack(side="right", fill="y", padx=(10, 20), pady=10)
        col.pack_propagate(False)
        self.themed.add(col, bg="BG")

        self.vocab_right_frame = tk.Frame(col, bg=p["BG"])
        self.vocab_right_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.themed.add(self.vocab_right_frame, bg="BG")
        self._build_vocab_right_column(self.vocab_right_frame)

        self.grammar_right_frame = tk.Frame(col, bg=p["BG"])
        self.grammar_right_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.themed.add(self.grammar_right_frame, bg="BG")
        self._build_grammar_right_column(self.grammar_right_frame)

        self.vocab_right_frame.tkraise()

    def _build_vocab_right_column(self, col):
        p = self.palette
        study_frame, study_widgets = build_study_summary_card(
            col, p, 0, self.open_study_mode, self.export_anki, themed=self.themed,
        )
        study_frame.pack(fill="x")
        self.readiness_book_label = study_widgets["book_label"]
        self.readiness_meta_label = study_widgets["meta_label"]
        self.readiness_label = study_widgets["readiness_label"]
        self.readiness_bar = study_widgets["progress_bar"]
        self.readiness_known_label = study_widgets["known_label"]
        self.readiness_to_learn_label = study_widgets["to_learn_label"]
        self.study_count_label = study_widgets["count_label"]
        self.study_flashcard_btn = study_widgets["flashcard_btn"]
        self.study_writing_btn = study_widgets["writing_btn"]
        self.study_anki_btn = study_widgets["anki_btn"]
        self.study_flashcard_btn.config(state="disabled")
        self.study_writing_btn.config(state="disabled")
        self.study_anki_btn.config(state="disabled")

        stats_frame = tk.Frame(
            col, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
        )
        stats_frame.pack(fill="both", expand=True, pady=(16, 0))
        self.themed.add(stats_frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        tk.Label(stats_frame, text="WORD FREQUENCY", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD).pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        self.chart_canvas = tk.Canvas(stats_frame, bg=p["PANEL"], highlightthickness=0, height=160)
        self.chart_canvas.pack(fill="x", padx=10)
        self.coverage_label = tk.Label(
            stats_frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT_SMALL_ITALIC,
            wraplength=230, justify="left",
        )
        self.coverage_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.themed.add(self.coverage_label, bg="PANEL", fg="DIM")
        # English/Spanish only (see readability_stats()'s docstring —
        # Flesch/Fernández Huerta's coefficients are each calibrated for
        # one specific language).
        self.readability_label = tk.Label(
            stats_frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT_SMALL_ITALIC,
            wraplength=230, justify="left",
        )
        self.readability_label.pack(anchor="w", padx=10, pady=(0, 10))
        self.themed.add(self.readability_label, bg="PANEL", fg="DIM")

    def _build_grammar_right_column(self, col):
        p = self.palette
        readiness_frame, readiness_widgets = build_grammar_readiness_card(
            col, p, self.open_grammar_quiz, self.export_grammar_anki, themed=self.themed,
        )
        readiness_frame.pack(fill="x")
        self.grammar_readiness_book_label = readiness_widgets["book_label"]
        self.grammar_readiness_meta_label = readiness_widgets["meta_label"]
        self.grammar_readiness_label = readiness_widgets["readiness_label"]
        self.grammar_readiness_bar = readiness_widgets["progress_bar"]
        self.grammar_readiness_known_label = readiness_widgets["known_label"]
        self.grammar_readiness_to_learn_label = readiness_widgets["to_learn_label"]
        self.grammar_quiz_count_label = readiness_widgets["count_label"]
        self.grammar_quiz_btn = readiness_widgets["quiz_btn"]
        self.grammar_identify_btn = readiness_widgets["identify_btn"]
        self.grammar_anki_btn = readiness_widgets["anki_btn"]
        self.grammar_quiz_btn.config(state="disabled")
        self.grammar_identify_btn.config(state="disabled")
        self.grammar_anki_btn.config(state="disabled")

        chart_frame = tk.Frame(
            col, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
        )
        chart_frame.pack(fill="both", expand=True, pady=(16, 0))
        self.themed.add(chart_frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        tk.Label(chart_frame, text="STRUCTURES BY CATEGORY", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD).pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        self.grammar_chart_canvas = tk.Canvas(chart_frame, bg=p["PANEL"], highlightthickness=0, height=160)
        self.grammar_chart_canvas.pack(fill="x", padx=10, pady=(0, 10))

    # ---- center: swaps between the Vocabulary and Grammar sections -----

    def _build_center(self, parent):
        # A single container holding both sections' content frames
        # stacked on top of each other via place() (each filling the
        # whole container) and swapped with .tkraise() — the standard Tk
        # idiom for two alternate full-size panels sharing one slot,
        # letting the sidebar/right-column stay exactly as they are.
        p = self.palette
        container = tk.Frame(parent, bg=p["BG"])
        container.pack(side="left", fill="both", expand=True, pady=10)
        self.themed.add(container, bg="BG")

        self.vocab_center_frame = tk.Frame(container, bg=p["BG"])
        self.vocab_center_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.themed.add(self.vocab_center_frame, bg="BG")
        self._build_vocab_center(self.vocab_center_frame)

        self.grammar_center_frame = tk.Frame(container, bg=p["BG"])
        self.grammar_center_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.themed.add(self.grammar_center_frame, bg="BG")
        grammar_frame, grammar_widgets = build_grammar_section(
            self.grammar_center_frame, p, self.themed, self._on_select_grammar_rule, self._grammar_mark,
        )
        grammar_frame.pack(fill="both", expand=True)
        self.grammar_tree = grammar_widgets["tree"]
        self.grammar_detail_header_var = grammar_widgets["detail_header_var"]
        self.grammar_detail_text = grammar_widgets["detail_text"]
        self.grammar_know_btn = grammar_widgets["know_btn"]
        self.grammar_learning_btn = grammar_widgets["learning_btn"]
        self._build_grammar_tree_context_menu()
        self.grammar_tree.bind("<Control-c>", self._copy_grammar_selection)
        self.grammar_tree.bind("<Control-C>", self._copy_grammar_selection)
        self.grammar_tree.bind("<Button-3>", self._show_grammar_tree_context_menu)
        self._clear_grammar_detail_panel()

        self.phrase_center_frame = tk.Frame(container, bg=p["BG"])
        self.phrase_center_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.themed.add(self.phrase_center_frame, bg="BG")
        phrase_frame, phrase_widgets = build_phrase_section(
            self.phrase_center_frame, p, self.themed, self._on_select_phrase,
            self.phrase_search_var,
        )
        phrase_frame.pack(fill="both", expand=True)
        self.phrase_tree = phrase_widgets["tree"]
        self.phrase_search_entry = phrase_widgets["search_entry"]
        self.phrase_detail_header_var = phrase_widgets["detail_header_var"]
        self.phrase_detail_text = phrase_widgets["detail_text"]
        self._build_phrase_tree_context_menu()
        self.phrase_tree.bind("<Control-c>", self._copy_phrase_selection)
        self.phrase_tree.bind("<Control-C>", self._copy_phrase_selection)
        self.phrase_tree.bind("<Button-3>", self._show_phrase_tree_context_menu)
        self._clear_phrase_detail_panel()

        self.vocab_center_frame.tkraise()

    def _build_vocab_center(self, parent):
        p = self.palette
        center = tk.Frame(parent, bg=p["BG"])
        center.pack(fill="both", expand=True)
        self.themed.add(center, bg="BG")

        controls = tk.Frame(center, bg=p["BG"])
        controls.pack(fill="x", pady=(0, 8))
        self.themed.add(controls, bg="BG")
        self.select_btn = ttk.Button(controls, text="SELECT BOOK", command=self.select_pdf, style="Accent.TButton")
        self.select_btn.pack(side="left")
        self.run_btn = ttk.Button(
            controls, text="GENERATE GLOSSARY", command=self.start_run, state="disabled", style="Accent.TButton",
        )
        self.run_btn.pack(side="right")
        self.stop_btn = ttk.Button(controls, text="STOP", command=self.stop_operation, state="disabled")
        self.stop_btn.pack(side="right", padx=(0, 10))
        self.file_label = tk.Label(controls, text="No file selected", bg=p["BG"], fg=p["DIM"], font=th.FONT, anchor="w")
        self.file_label.pack(side="left", fill="x", expand=True, padx=15)
        self.themed.add(self.file_label, bg="BG", fg="DIM")

        lang_frame = tk.Frame(center, bg=p["BG"])
        lang_frame.pack(fill="x", pady=(0, 8))
        self.themed.add(lang_frame, bg="BG")
        lang_hdr = tk.Label(lang_frame, text="BOOK LANGUAGE", bg=p["BG"], fg=p["FG"], font=th.FONT_BOLD)
        lang_hdr.pack(side="left")
        self.themed.add(lang_hdr, bg="BG", fg="FG")
        # A dropdown, not a row of radio buttons: at a correctly
        # DPI-scaled font size, 5 radio buttons (one label reading
        # "Arabic (Qur'an language)") no longer fit next to the sidebar
        # and right column at any reasonable window width — confirmed
        # directly, "German" was being pushed off-screen entirely.
        self._lang_label_to_code = {label: code for code, label, _iso in SELECTABLE_LANGUAGES}
        self.lang_combo = ttk.Combobox(
            lang_frame, values=[label for _c, label, _i in SELECTABLE_LANGUAGES],
            state="readonly", font=th.FONT, width=26,
        )
        self.lang_combo.current(0)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_combo_change)
        self.lang_combo.pack(side="left", padx=(12, 0))

        options = tk.Frame(center, bg=p["BG"])
        options.pack(fill="x", pady=(0, 8))
        self.themed.add(options, bg="BG")
        self.hide_known_check = ttk.Checkbutton(
            options, text="Hide words I already know", variable=self.hide_known_var, command=self.render_results,
        )
        self.hide_known_check.pack(side="left")
        # English only: word-family grouping relies on WordNet's `morphy`
        # analyzer, which has no equivalent for the app's other supported
        # languages — see word_family_root()'s docstring in
        # text_analyzer.py. Disabled (not hidden) for other languages so
        # its existence is still discoverable.
        self.group_families_check = ttk.Checkbutton(
            options, text="Group word families (book/books/booked)",
            variable=self.group_families_var, command=self.render_results, state="disabled",
        )
        self.group_families_check.pack(side="left", padx=(16, 0))

        search_frame = tk.Frame(center, bg=p["BG"])
        search_frame.pack(fill="x", pady=(0, 8))
        self.themed.add(search_frame, bg="BG")
        search_hdr = tk.Label(search_frame, text="FILTER", bg=p["BG"], fg=p["FG"], font=th.FONT_BOLD)
        search_hdr.pack(side="left")
        self.themed.add(search_hdr, bg="BG", fg="FG")
        self.sort_btn = ttk.Button(search_frame, text="SORT: FREQUENCY", command=self.toggle_sort)
        self.sort_btn.pack(side="right")
        self.level_filter_combo = ttk.Combobox(
            search_frame, textvariable=self.level_filter_var,
            values=("All levels", "A1", "A2", "B1", "B2", "C1", "C2", "No level"),
            state="readonly", font=th.FONT, width=10,
        )
        self.level_filter_combo.bind("<<ComboboxSelected>>", lambda _e: self.render_results())
        self.level_filter_combo.pack(side="right", padx=(0, 10))
        self.search_entry = tk.Entry(
            search_frame, textvariable=self.search_var, bg=p["PANEL"], fg=p["FG"], insertbackground=p["FG"],
            font=th.FONT, relief="flat", highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
            highlightcolor=p["ACCENT"],
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=10, ipady=4)
        self.themed.add(
            self.search_entry, bg="PANEL", fg="FG", insertbackground="FG",
            highlightbackground="PANEL_BORDER", highlightcolor="ACCENT",
        )

        self.progress = ttk.Progressbar(center, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 5))

        # Word table + detail panel, side by side, both expanding.
        table_area = tk.Frame(center, bg=p["BG"])
        table_area.pack(fill="both", expand=True)
        self.themed.add(table_area, bg="BG")

        tree_frame = tk.Frame(table_area, bg=p["BG"])
        tree_frame.pack(side="top", fill="both", expand=True)
        self.themed.add(tree_frame, bg="BG")
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self.tree = ttk.Treeview(
            tree_frame, columns=("count", "pos", "cefr", "definition"), show="tree headings",
            yscrollcommand=scrollbar.set, selectmode="extended",
        )
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.tree.yview)
        self.tree.heading("#0", text="Word")
        self.tree.heading("count", text="Count")
        self.tree.heading("pos", text="Part of Speech")
        self.tree.heading("cefr", text="Level")
        self.tree.heading("definition", text="Definition")
        self.tree.column("#0", width=220, anchor="w")
        self.tree.column("count", width=70, anchor="center")
        # Blank (not "—") for a word with no part-of-speech/CEFR data at
        # all — every offline-dictionary-only language for POS (see
        # build_pos_map()'s docstring), every language but English/
        # Spanish for CEFR (see build_cefr_map()'s docstring) — rather
        # than a dash implying something was looked up and came back
        # empty, when nothing was ever looked up there.
        self.tree.column("pos", width=110, anchor="center")
        self.tree.column("cefr", width=60, anchor="center")
        self.tree.column("definition", width=340, anchor="w")
        self.tree.tag_configure("new", foreground=p["ACCENT"], font=th.FONT_ROW_WORD)
        self.tree.tag_configure("learning", foreground=p["SECOND_FG"], font=th.FONT_ROW_WORD)
        self.tree.tag_configure("known", foreground=p["DIM"], font=th.FONT)
        self.tree.tag_configure("family", foreground=p["DIM"], font=th.FONT_SMALL_BOLD)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_word)
        # A plain click toggles that one row in/out of the selection —
        # no Ctrl/Shift needed — so clicking through several words in a
        # row builds up a multi-selection directly, then a single I KNOW
        # IT / LEARNING click applies to all of them. This replaces
        # ttk's own default click handling (which would instead replace
        # the whole selection with just the clicked row); Shift+click
        # still does its normal range-select since that's a different
        # event and isn't intercepted here.
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<Control-c>", self._copy_selection)
        self.tree.bind("<Control-C>", self._copy_selection)
        self._build_tree_context_menu()
        self.tree.bind("<Button-3>", self._show_tree_context_menu)

        self._build_detail_panel(table_area)

    def _build_tree_context_menu(self):
        p = self.palette
        self.tree_menu = tk.Menu(
            self, tearoff=0, bg=p["PANEL"], fg=p["FG"], activebackground=p["SECOND_HOVER"],
            activeforeground=p["FG"], font=th.FONT,
        )
        self.tree_menu.add_command(label="✅ Mark as known", command=lambda: self._detail_mark("known"))
        self.tree_menu.add_command(label="📖 Mark as still learning", command=lambda: self._detail_mark("learning"))
        self.tree_menu.add_separator()
        # A Treeview can't be click-dragged to select a run of characters
        # the way real text can (confirmed: this came up directly as "I
        # can't select anything to copy") — these two give a one-click
        # way to get exactly the word, or exactly the definition, onto
        # the clipboard without needing that. Copy (Ctrl+C) below still
        # copies everything (word/count/definition/example) tab-
        # separated, for pasting into a spreadsheet.
        self.tree_menu.add_command(label="Copy word", command=self._copy_words_only)
        self.tree_menu.add_command(label="Copy definition", command=self._copy_definitions_only)
        self.tree_menu.add_command(label="Copy (Ctrl+C)", command=self._copy_selection)
        self.themed.add(
            self.tree_menu, bg="PANEL", fg="FG", activebackground="SECOND_HOVER", activeforeground="FG",
        )

    def _on_tree_click(self, event):
        # Column headers, the empty area below the last row, etc. have no
        # row under the cursor — let those fall through to ttk's own
        # default handling (e.g. clicking empty space still clears the
        # selection normally) rather than swallowing every click.
        region = self.tree.identify_region(event.x, event.y)
        if region not in ("tree", "cell"):
            return
        # A family group's expand/collapse triangle lives in this same
        # "tree" region — without this check, clicking it would toggle
        # the row into the selection instead of expanding it, and
        # `return "break"` below would also block ttk's own toggle
        # handler from ever running at all.
        if self.tree.identify_element(event.x, event.y) == "Treeitem.indicator":
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        current = list(self.tree.selection())
        if row in current:
            current.remove(row)
        else:
            current.append(row)
        # selection_set() fires <<TreeviewSelect>> on its own, updating
        # the Detail panel — nothing else needed here. Returning "break"
        # stops ttk's own click handler from also running and collapsing
        # the selection back down to just this one row.
        self.tree.selection_set(current)
        return "break"

    def _show_tree_context_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row and row not in self.tree.selection():
            # Right-clicking a row outside the current selection replaces
            # it, matching how File Explorer / most row-based UIs behave
            # — right-clicking inside an existing multi-selection instead
            # keeps it, so the menu acts on everything already selected.
            self.tree.selection_set(row)
        if not self.tree.selection():
            return
        self.tree_menu.tk_popup(event.x_root, event.y_root)

    def _on_lang_combo_change(self, _event):
        label = self.lang_combo.get()
        self.lang_choice_var.set(self._lang_label_to_code.get(label, "auto"))

    def _build_detail_panel(self, parent):
        p = self.palette
        panel = tk.Frame(parent, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
        panel.pack(side="bottom", fill="x", pady=(10, 0))
        self.themed.add(panel, bg="PANEL", highlightbackground="PANEL_BORDER")

        # A readonly Entry, not a Label: state="readonly" blocks typing
        # but — same trick as detail_text below — still allows mouse
        # drag-to-select and Ctrl+C/right-click Copy, which a Label can't
        # do at all. The word itself is exactly what a reader most often
        # wants to copy (e.g. to paste into a dictionary/translator
        # elsewhere), so it needed the same selectability the definition
        # text already had.
        self.detail_word_var = tk.StringVar(value="Select a word to see details")
        self.detail_word_entry = tk.Entry(
            panel, textvariable=self.detail_word_var, state="readonly",
            readonlybackground=p["PANEL"], fg=p["ACCENT"], font=th.FONT_ROW_WORD,
            relief="flat", highlightthickness=0, bd=0, insertbackground=p["ACCENT"],
            selectbackground=p["ACCENT"], selectforeground=p["ACCENT_TEXT"],
            cursor="xterm",
        )
        self.detail_word_entry.pack(fill="x", padx=14, pady=(10, 2))
        self.themed.add(
            self.detail_word_entry, bg="PANEL", fg="ACCENT", readonlybackground="PANEL",
            selectbackground="ACCENT", selectforeground="ACCENT_TEXT",
        )

        # A Text widget, not Labels: state="disabled" blocks typing/edits
        # but Tk still allows mouse drag-to-select and Ctrl+C/right-click
        # Copy inside a disabled Text widget — a Label can't be selected
        # at all, so this is what actually lets a reader highlight a
        # definition (or just part of one) with the cursor and copy it.
        self.detail_text = tk.Text(
            panel, bg=p["PANEL"], fg=p["FG"], font=th.FONT, wrap="word", relief="flat",
            height=4, highlightthickness=0, padx=0, pady=0, borderwidth=0,
            cursor="xterm", insertbackground=p["FG"],
        )
        self.detail_text.pack(fill="x", padx=14)
        self.detail_text.tag_configure("def", foreground=p["FG"], font=th.FONT)
        self.detail_text.tag_configure("example", foreground=p["RUBRIC"], font=th.FONT_SMALL_ITALIC)
        self.detail_text.tag_configure("hint", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
        self.detail_text.config(state="disabled")
        self.themed.add(self.detail_text, bg="PANEL", fg="FG", insertbackground="FG")

        btn_row = tk.Frame(panel, bg=p["PANEL"])
        btn_row.pack(fill="x", padx=14, pady=10)
        self.themed.add(btn_row, bg="PANEL")
        self.detail_speak_btn = ttk.Button(btn_row, text="\U0001F50A SPEAK", command=self._detail_speak, state="disabled")
        self.detail_speak_btn.pack(side="left")
        self.detail_find_btn = ttk.Button(btn_row, text="\U0001F50E FIND ALL", command=self._detail_find_all, state="disabled")
        self.detail_find_btn.pack(side="left", padx=(8, 0))
        self.detail_word_info_btn = ttk.Button(
            btn_row, text="\U0001F52C WORD INFO", command=self._detail_word_info, state="disabled",
        )
        self.detail_word_info_btn.pack(side="left", padx=(8, 0))
        # Opposite enabled-state from the three buttons above: those are
        # single-word-only and disable on a multi-selection; this one
        # only makes sense FOR a multi-selection (a single word already
        # has the full detail panel to itself) — see _show_selection().
        self.detail_focus_btn = ttk.Button(
            btn_row, text="\U0001F50D FOCUS VIEW", command=self._detail_focus_view, state="disabled",
        )
        self.detail_focus_btn.pack(side="left", padx=(8, 0))
        self.detail_know_btn = ttk.Button(btn_row, text="✅ I KNOW IT", command=lambda: self._detail_mark("known"), state="disabled")
        self.detail_know_btn.pack(side="left", padx=(8, 0))
        self.detail_learning_btn = ttk.Button(btn_row, text="\U0001F4D6 LEARNING", command=lambda: self._detail_mark("learning"), state="disabled")
        self.detail_learning_btn.pack(side="left", padx=(8, 0))

    def _build_statusbar(self):
        p = self.palette
        bar = tk.Frame(self, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
        bar.pack(side="bottom", fill="x")
        self.themed.add(bar, bg="PANEL", highlightbackground="PANEL_BORDER")
        self.status_label = tk.Label(
            bar, text="Select a PDF or EPUB to begin.", bg=p["PANEL"], fg=p["DIM"], font=th.FONT,
            anchor="w", justify="left",
        )
        self.status_label.pack(side="left", fill="x", expand=True, padx=14, pady=6)
        self.themed.add(self.status_label, bg="PANEL", fg="DIM")
        self.elapsed_label = tk.Label(bar, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT)
        self.elapsed_label.pack(side="right", padx=14, pady=6)
        self.themed.add(self.elapsed_label, bg="PANEL", fg="DIM")
        self.bind("<Configure>", self._on_resize_status_label)

        # Footer actions: STUDY (full flashcard window) + SAVE AS / export.
        footer = tk.Frame(self, bg=p["BG"])
        footer.pack(side="bottom", fill="x", padx=20, pady=(0, 10))
        self.themed.add(footer, bg="BG")
        self.save_btn = ttk.Button(footer, text="SAVE AS... (Ctrl+S)", command=self.save_as, state="disabled")
        self.save_btn.pack(side="right")
        self.anki_btn = ttk.Button(footer, text="EXPORT TO ANKI...", command=self.export_anki, state="disabled")
        self.anki_btn.pack(side="right", padx=(0, 8))
        self.concordance_btn = ttk.Button(
            footer, text="\U0001F50E FIND IN BOOK...", command=self.open_concordance, state="disabled",
        )
        self.concordance_btn.pack(side="left")

        self.bind("<Control-o>", lambda _e: self.select_pdf())
        self.bind("<Control-O>", lambda _e: self.select_pdf())
        self.bind("<Control-f>", lambda _e: self._focus_search())
        self.bind("<Control-F>", lambda _e: self._focus_search())
        self.bind("<Control-s>", lambda _e: self.save_as())
        self.bind("<Control-S>", lambda _e: self.save_as())
        self.bind("<Control-e>", lambda _e: self.save_as())
        self.bind("<Control-E>", lambda _e: self.save_as())
        self.bind("<Escape>", lambda _e: self._clear_search())

    def _focus_search(self):
        # Same section-aware reuse as save_as()/export_anki() — Ctrl+F
        # focuses whichever section's own search box is actually
        # visible right now, not always the vocabulary one. Grammar has
        # no search box of its own (its whole catalogue is at most ~28
        # rows, already grouped into 7 categories — a filter box adds
        # no real value there, see build_grammar_section()'s docstring
        # for that same "don't add UI weight a short list doesn't need"
        # call), so it's left out of this branch on purpose.
        if self.active_section == "phrases":
            self.phrase_search_entry.focus_set()
        else:
            self.search_entry.focus_set()

    def _clear_search(self):
        if self.active_section == "phrases":
            self.phrase_search_var.set("")
        else:
            self.search_var.set("")

    def _on_resize_status_label(self, event):
        if event.widget is self:
            new_width = max(200, event.width - 260)
            if abs(new_width - self.status_label.cget("wraplength")) > 4:
                self.status_label.config(wraplength=new_width)

    # ------------------------------------------------------------- theme

    def toggle_theme(self):
        self.settings["theme"] = "dark" if self.palette is th.LIGHT else "light"
        save_settings(self.settings)
        self.palette = th.PALETTES[self.settings["theme"]]
        style = ttk.Style(self)
        th.build_ttk_style(style, self.palette)
        self.themed.refresh(self.palette)
        self.theme_btn.config(text=self._theme_icon())
        p = self.palette
        # The root Tk window's own background was only ever set once, at
        # construction — never updated here, which is exactly why toggling
        # left part of the page still showing the old theme's color
        # (anywhere the packed frames don't fully cover the window, e.g.
        # during a resize). Every other themed widget lives inside frames
        # that ARE covered by the registry below; the root itself isn't.
        self.configure(bg=p["BG"])
        self.tree.tag_configure("new", foreground=p["ACCENT"])
        self.tree.tag_configure("learning", foreground=p["SECOND_FG"])
        self.tree.tag_configure("known", foreground=p["DIM"])
        self.tree.tag_configure("family", foreground=p["DIM"])
        self.detail_text.tag_configure("def", foreground=p["FG"])
        self.detail_text.tag_configure("example", foreground=p["RUBRIC"])
        self.detail_text.tag_configure("hint", foreground=p["DIM"])
        self.grammar_tree.tag_configure("category", foreground=p["ACCENT"])
        self.grammar_tree.tag_configure("new", foreground=p["ACCENT"])
        self.grammar_tree.tag_configure("learning", foreground=p["SECOND_FG"])
        self.grammar_tree.tag_configure("known", foreground=p["DIM"])
        self.grammar_detail_text.tag_configure("def", foreground=p["FG"])
        self.grammar_detail_text.tag_configure("rule", foreground=p["DIM"])
        self.grammar_detail_text.tag_configure("num", foreground=p["DIM"])
        self.grammar_detail_text.tag_configure("example", foreground=p["RUBRIC"])
        self.grammar_detail_text.tag_configure("hint", foreground=p["DIM"])
        self.chart_canvas.config(bg=p["PANEL"])
        self._draw_chart()
        self.grammar_chart_canvas.config(bg=p["PANEL"])
        self._draw_grammar_chart()
        th.apply_dark_titlebar(self, self.settings["theme"] == "dark")

    def _show_settings(self):
        messagebox.showinfo(
            "Settings",
            "Preferences are saved automatically.\n\n"
            f"Theme: {self.settings.get('theme', 'light').title()}\n"
            "Use RESET PROGRESS in the sidebar to clear all known/learning history.",
        )

    def _show_help(self):
        messagebox.showinfo(
            "Keyboard shortcuts",
            "Ctrl+O   Open a book\n"
            "Ctrl+F   Focus the filter box\n"
            "Ctrl+S / Ctrl+E   Save / export\n"
            "Esc      Clear the filter box\n\n"
            "In Study mode (Flashcards):\n"
            "Space    Reveal the definition\n"
            "1        I know it\n"
            "2        Still learning\n"
            "S        Speak the word\n\n"
            "In Study mode (Writing Quiz):\n"
            "Enter    Check your answer, then move to the next word\n"
            "S        Speak the word (after checking)\n\n"
            "Tip: for English books, \"Group word families\" collapses "
            "book/books/booked into one row, and EXPORT TO ANKI writes a "
            "ready-to-import flashcard deck. FIND IN BOOK (or FIND ALL in "
            "the Detail panel) shows every sentence a word or phrase "
            "appears in, not just one example. Click 🎓 above to replay "
            "the full tutorial.",
        )

    def show_onboarding(self):
        def mark_done():
            self.settings["onboarded"] = True
            save_settings(self.settings)

        OnboardingWizard(self, self.palette, on_done=mark_done)

    # ---------------------------------------------------------------- select

    def select_pdf(self):
        path = filedialog.askopenfilename(
            title="Select a PDF or EPUB",
            filetypes=[
                ("Books (PDF/EPUB)", "*.pdf;*.epub"),
                ("PDF files", "*.pdf"),
                ("EPUB files", "*.epub"),
            ],
        )
        if path:
            self._select_book_path(path)

    def _select_book_path(self, path):
        self.filepath = path
        self.book_text = None
        self.detected_lang = None
        name = path.split("/")[-1].split("\\")[-1]
        self.file_label.config(text=name if len(name) <= 55 else name[:52] + "...")
        self.run_btn.config(state="normal")
        self.settings = add_recent_book(self.settings, path, name)
        save_settings(self.settings)
        self._refresh_recent_books()

    # ------------------------------------------------------------ glossary

    def start_run(self):
        if not self.filepath:
            return
        self.cancel_event.clear()
        self._set_busy(True)
        self.glossary_entries = []
        self.undefined_words = []
        self.word_freqs = {}
        self.word_examples = {}
        self.word_pos_map = {}
        self.word_cefr_map = {}
        self.word_lookup = {}
        self.family_map = {}
        self.word_family_root_of = {}
        self.family_definition = {}
        self.family_pos = {}
        self.family_cefr = {}
        self.book_sentences = []
        self.grammar_results = {}
        self.phrases = {}
        self.render_results()
        self._render_grammar_results()
        self._render_phrases()
        self.status_label.config(text="Reading book...")
        self.progress.config(mode="indeterminate")
        self.progress.start(15)
        self.run_start_time = time.monotonic()

        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def stop_operation(self):
        self.cancel_event.set()
        self.status_label.config(text="Stopping...")
        self.stop_btn.config(state="disabled")

    def _keep_going(self):
        return not self.cancel_event.is_set()

    def _get_book_text(self):
        if self.book_text is None:
            self.book_text = load_book_text(self.filepath)
        return self.book_text

    def _switch_active_language(self, lang):
        """Point self.known_words/learning_words/progress_schedule/
        progress_learned_at at `lang`'s own progress bucket in
        self.all_progress — creating an empty one if this language has
        no progress yet. Called once at startup (defaulting to "en"
        before any book's language is known) and again every time a
        glossary finishes generating, so opening a German book after an
        English one switches to (and only ever affects) German progress,
        never mixing the two. Safe to call with the language already
        active — it's a no-op re-point to the same bucket.
        """
        lang = lang or "en"
        if lang not in self.all_progress:
            self.all_progress[lang] = empty_language_progress()
        bucket = self.all_progress[lang]
        self.active_lang = lang
        self.known_words = bucket["known"]
        self.learning_words = bucket["learning"]
        self.progress_schedule = bucket["schedule"]
        self.progress_learned_at = bucket["learned_at"]

    def _save_progress(self):
        # Every save writes the WHOLE progress file (every language plus
        # grammar) at once — save_progress() has no partial-write mode,
        # so any call site that saves must pass every bucket it isn't
        # itself changing too, or that bucket would silently be wiped to
        # empty on disk. One shared helper means a new call site can
        # never forget the grammar bucket the way a hand-written call
        # easily could.
        save_progress(self.all_progress, self.reset_backups, {
            "known": self.grammar_known, "learning": self.grammar_learning,
            "schedule": self.grammar_schedule, "learned_at": self.grammar_learned_at,
        })

    def _refresh_vocab_stat(self):
        # No language name on this header or the RESET button (tried
        # both, removed by request — appending it made them too wide for
        # the sidebar's fixed width and it visibly overflowed/clipped,
        # same problem in both spots). The button's own confirmation
        # dialog (see reset_progress()) still says which language it's
        # about to clear — that's the one place removing it would
        # actually lose information, so it stays there. The PROGRESS
        # card below and the READING READINESS card in the right column
        # both still name the active language too.
        self.vocab_header_label.config(text="YOUR VOCABULARY")
        self.reset_btn.config(text="RESET PROGRESS")
        known, learning = len(self.known_words), len(self.learning_words)
        if known or learning:
            text = f"{known} known\n{learning} still learning\n(across all books)"
        else:
            text = "None yet — use STUDY to start building your vocabulary."
        self.vocab_stat_label.config(text=text)

    def _refresh_reading_readiness(self):
        """Update the READING READINESS card with the currently loaded
        book's own numbers. "Readiness" is the share of this book's
        DEFINED vocabulary (glossary_entries — the words Study mode can
        actually quiz on) already marked known, not the raw unique-word
        count, which also includes undefined words (names, typos, words
        no dictionary matched) that can't be marked known/learning at
        all — counting those against readiness would make it permanently
        unreachable for a book with a lot of them, which isn't honest.
        """
        if not self.glossary_entries and not self.undefined_words:
            self.readiness_book_label.config(text="No book loaded yet.")
            self.readiness_meta_label.config(text="")
            self.readiness_label.config(text="")
            self.readiness_bar.config(value=0)
            self.readiness_known_label.config(text="")
            self.readiness_to_learn_label.config(text="")
            return

        title = "?"
        if self.filepath:
            title = os.path.splitext(os.path.basename(self.filepath))[0]
            if len(title) > 34:
                title = title[:31] + "..."
        self.readiness_book_label.config(text=title)

        total_unique = len(self.glossary_entries) + len(self.undefined_words)
        lang_name = language_name(self.detected_lang) if self.detected_lang else "?"
        self.readiness_meta_label.config(text=f"{lang_name} · {total_unique:,} unique words")

        defined_total = len(self.glossary_entries)
        known_in_book = sum(1 for w, _ in self.glossary_entries if w in self.known_words)
        to_learn = defined_total - known_in_book
        pct = round(known_in_book / defined_total * 100) if defined_total else 0

        self.readiness_label.config(text=f"📊 Reading readiness: {pct}%")
        self.readiness_bar.config(value=pct)
        self.readiness_known_label.config(text=f"✅ Known: {known_in_book:,} words")
        self.readiness_to_learn_label.config(text=f"📖 To learn: {to_learn:,} words")

    def _refresh_progress_stat(self):
        """The sidebar's PROGRESS card: how much vocabulary growth has
        happened recently (needs learned_at — see mark_known()'s
        docstring for why words known from before it existed don't count
        here), plus the current book's goal and how close "comfortable
        reading" (a common 95%-of-vocabulary benchmark in reading
        research) is. Everything here is scoped to self.active_lang —
        known_words/progress_learned_at are already that language's own
        live bucket (see _switch_active_language()), so a German book's
        progress can never inflate an English count or vice versa; the
        header names the language so that scoping is visible, not just
        true underneath.
        """
        lang_name = language_name(self.active_lang)
        self.progress_header_label.config(text=f"PROGRESS — {lang_name.upper()}")
        this_week = words_learned_since(self.progress_learned_at, 7)
        this_month = words_learned_since(self.progress_learned_at, 30)
        lines = [
            f"📅 This week: {this_week} word(s) learned",
            f"📅 This month: {this_month} word(s) learned",
            f"📅 All time: {len(self.known_words):,} word(s) known",
        ]
        # English/Spanish only (see estimate_vocabulary_level()'s
        # docstring) — a rough proxy from known-word coverage of the
        # bundled CEFR data, not a real placement-test result, so it's
        # worded as an estimate rather than a bare label.
        if self.active_lang in ("en", "es"):
            level = estimate_vocabulary_level(self.known_words, self.active_lang)
            lines.append(f"🎓 Estimated level: {level or 'Beginner (pre-A1)'}")
        if self.glossary_entries:
            defined_total = len(self.glossary_entries)
            known_in_book = sum(1 for w, _ in self.glossary_entries if w in self.known_words)
            pct = round(known_in_book / defined_total * 100) if defined_total else 0
            target = 95
            lines.append("")
            lines.append(f"🎯 Goal: {defined_total:,} word(s) in this book ({pct}% there)")
            if pct >= target:
                lines.append(f"📖 Reading readiness: {pct}% — comfortable reading level reached!")
            else:
                lines.append(f"📖 Reading readiness: {pct}% ({target - pct}% to a comfortable {target}%)")
        self.progress_stat_label.config(text="\n".join(lines))
        self._refresh_restore_button()

    def _refresh_restore_button(self):
        backup = restorable_backup(self.reset_backups, self.active_lang)
        self.restore_btn.config(state="normal" if backup else "disabled")

    def _record_study_results(self, known_words, learning_words):
        # self.progress_schedule and self.progress_learned_at were both
        # already updated in place during the session (StudyWindow calls
        # review_word()/mark_known() per card), so they just need to be
        # saved here alongside known/learning, not merged.
        self.known_words |= set(known_words)
        self.learning_words |= set(learning_words)
        self.learning_words -= self.known_words
        self._save_progress()
        self._refresh_vocab_stat()
        self._refresh_reading_readiness()
        self._refresh_progress_stat()
        self.render_results()

    def reset_progress(self):
        if not self.known_words and not self.learning_words:
            return
        lang_name = language_name(self.active_lang)
        if messagebox.askyesno(
            "Reset progress",
            f"Clear your {lang_name} word history (known + still-learning)? Other languages' progress is "
            f"untouched.\n\nYou can undo this from RESTORE PROGRESS for up to {RESET_BACKUP_DAYS} days.",
        ):
            # Snapshot before clearing, so RESTORE PROGRESS can bring it
            # back — see backup_before_reset()'s docstring. Cleared with
            # .clear() rather than reassigning to a new set/dict: these
            # four names are live references into self.all_progress (and
            # possibly into an already-open StudyWindow too), so a
            # reassignment here would silently stop updating either.
            backup_before_reset(
                self.reset_backups, self.active_lang,
                self.known_words, self.learning_words, self.progress_schedule, self.progress_learned_at,
            )
            self.known_words.clear()
            self.learning_words.clear()
            self.progress_schedule.clear()
            self.progress_learned_at.clear()
            self._save_progress()
            self._refresh_vocab_stat()
            self._refresh_reading_readiness()
            self._refresh_progress_stat()
            self.render_results()

    def restore_progress(self):
        backup = restorable_backup(self.reset_backups, self.active_lang)
        if not backup:
            return
        lang_name = language_name(self.active_lang)
        if messagebox.askyesno(
            "Restore progress",
            f"Restore your {lang_name} progress from just before the last reset "
            f"({len(backup['known'])} known, {len(backup['learning'])} still learning)?",
        ):
            # Same in-place-mutation reasoning as reset_progress() above
            # — these are live references, not just local variables.
            self.known_words.clear()
            self.known_words.update(backup["known"])
            self.learning_words.clear()
            self.learning_words.update(backup["learning"])
            self.progress_schedule.clear()
            self.progress_schedule.update(backup["schedule"])
            self.progress_learned_at.clear()
            self.progress_learned_at.update(backup["learned_at"])
            # One-time use: this backup is tied to a specific reset, and
            # restoring it resolves that reset — a later reset makes its
            # own fresh backup rather than this one lingering ambiguously.
            del self.reset_backups[self.active_lang]
            self._save_progress()
            self._refresh_vocab_stat()
            self._refresh_reading_readiness()
            self._refresh_progress_stat()
            self.render_results()

    def run_pipeline(self):
        try:
            text = self._get_book_text()
            self.work_queue.put(("status", "Extracting unique words..."))
            freqs = word_frequencies(text)
            words = sorted(freqs.keys())
            choice = self.lang_choice_var.get()
            lang = detect_language(text) if choice == "auto" else choice
            wordnet_lang = WORDNET_LANG_CODES.get(lang)
            offline_dict_lang = lang if lang in OFFLINE_DICT_LANGS else None
            # Split once, reused for both this run's example sentences and
            # any later Find in Book search — splitting is the expensive
            # part, not the search itself.
            sentences = split_sentences(text)
            examples = build_example_sentences(text, words, sentences=sentences)
            self.work_queue.put(("status", f"Found {len(words)} unique words. Checking dictionary data..."))
            ensure_wordnet()
            self.work_queue.put(("progress_setup", len(words)))

            last_status_at = [0.0]

            def on_progress(i, total):
                self.work_queue.put(("progress", i, total))
                now = time.monotonic()
                if now - last_status_at[0] > 0.5:
                    last_status_at[0] = now
                    self.work_queue.put(("status", f"Looking up definitions... {i}/{total} words"))

            defined, undefined, _ = build_glossary(
                words, on_progress=on_progress, should_continue=self._keep_going,
                wordnet_lang=wordnet_lang, offline_dict_lang=offline_dict_lang,
            )
            # WordNet-only (English/Spanish/French/Arabic-via-WordNet) —
            # German/Turkish/Russian/most-Arabic simply get {} here since
            # their offline dictionaries carry no part-of-speech data at
            # all (see build_pos_map()'s docstring).
            pos_map = build_pos_map([w for w, _d in defined], wordnet_lang=wordnet_lang)
            # English/Spanish only (see build_cefr_map()'s docstring) —
            # every other language's `defined` list simply yields {} here.
            cefr_map = build_cefr_map([w for w, _d in defined], lang=lang, pos_map=pos_map)
            # Grammar analysis is the last step of this same run rather
            # than a separate button/thread: tagging every sentence with
            # nltk's lightweight perceptron tagger is expected to be in
            # the same low-single-digit-to-tens-of-seconds range this
            # step's dictionary-lookup loop above already runs in, and a
            # second on-demand action would split "analyze this book"
            # into two separate waits/cancel-paths for little benefit.
            # English/Spanish only (see analyze_grammar()'s docstring) —
            # every other language gets {} for free. Only English needs
            # nltk's tagger downloaded first; Spanish detection works
            # from spelling/suffixes alone, no tagger involved.
            if lang in ("en", "es"):
                self.work_queue.put(("status", "Analyzing grammar structures..."))
            if lang == "en":
                ensure_pos_tagger()
            grammar_results = analyze_grammar(sentences, lang=lang, should_continue=self._keep_going)
            # English/Spanish only (see readability_stats()'s docstring)
            # — every other language gets None, same "quietly absent"
            # convention as pos_map/cefr_map/grammar_results.
            readability = readability_stats(text, lang=lang)
            # Language-general (see extract_phrases()'s docstring) —
            # unlike grammar/CEFR/pronunciation, this isn't English-only,
            # so every book gets phrase extraction, not just English/
            # Spanish ones (though only those two get the "recognized"
            # real-dictionary tier — every language still gets "recurring").
            ensure_phrase_data(lang)
            phrases = extract_phrases(sentences, lang=lang)
            self.work_queue.put((
                "glossary_done", defined, undefined, lang, freqs, examples, sentences, pos_map, cefr_map,
                grammar_results, readability, phrases,
            ))
        except Exception as exc:
            # A bare str(exc) alone ("sequence item 1: expected str
            # instance, NoneType found" and the like) names WHAT went
            # wrong but not WHERE — confirmed directly this matters: a
            # real report of exactly this shape couldn't be pinned down
            # from the message text alone, across extensive attempted
            # reproduction, for lack of a file/line to look at. The full
            # traceback is written to error_log.txt next to the .exe
            # (appended, not overwritten, so more than one incident
            # stays available) — app_dir(), not _resource_dir(), since
            # this needs to be a real, persistent, user-findable file,
            # not something bundled read-only into the frozen archive.
            try:
                log_path = os.path.join(app_dir(), "error_log.txt")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{'=' * 80}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(traceback.format_exc())
            except OSError:
                log_path = None
            message = str(exc)
            if log_path:
                message += f" (full details in {os.path.basename(log_path)})"
            self.work_queue.put(("error", message))

    # ------------------------------------------------------------- polling

    def poll_queue(self):
        try:
            while True:
                item = self.work_queue.get_nowait()
                kind = item[0]
                if kind == "status":
                    self.status_label.config(text=item[1])
                elif kind == "progress_setup":
                    self.progress.stop()
                    self.progress.config(mode="determinate", maximum=item[1], value=0)
                elif kind == "progress":
                    self.progress.config(value=item[1])
                elif kind == "glossary_done":
                    (
                        _, defined, undefined, lang, freqs, examples, sentences, pos_map, cefr_map,
                        grammar_results, readability, phrases,
                    ) = item
                    self.glossary_entries = defined
                    self.undefined_words = undefined
                    self.detected_lang = lang
                    # Switches self.known_words/learning_words/
                    # progress_schedule/progress_learned_at to (and, the
                    # first time, creates) this language's own progress
                    # bucket — before render_results() below, so word
                    # statuses in the table reflect the right language's
                    # history rather than whatever was active a moment
                    # ago from a previous book.
                    self._switch_active_language(lang)
                    self._refresh_vocab_stat()
                    self._refresh_restore_button()
                    self.word_freqs = freqs
                    self.word_examples = examples
                    self.book_sentences = sentences
                    self.word_pos_map = pos_map
                    self.word_cefr_map = cefr_map
                    self.grammar_results = grammar_results
                    self.phrases = phrases
                    self._build_families()
                    self.render_results()
                    self._render_grammar_results()
                    self._render_phrases()
                    self._draw_chart()
                    self._draw_grammar_chart()
                    self._refresh_grammar_stat()
                    self._refresh_grammar_readiness()
                    self._refresh_grammar_progress_stat()
                    stopped = self.cancel_event.is_set()
                    prefix = "Stopped — " if stopped else "Done — "
                    text = f"{prefix}{len(defined)} defined, {len(undefined)} without a dictionary match."
                    if not stopped and lang != "en" and undefined and not defined:
                        text += f" ({language_name(lang)} detected — no dictionary matches were found for this book.)"
                    cov = exp.coverage_line(freqs)
                    if not stopped and cov:
                        text += f" {cov}"
                    self.status_label.config(text=text)
                    self.coverage_label.config(text=cov or "")
                    if readability:
                        formula_name = "Fernández Huerta" if lang == "es" else "Flesch"
                        self.readability_label.config(text=(
                            f"Avg. sentence length: {readability['avg_sentence_length']} words   ·   "
                            f"Lexical density: {readability['lexical_density']}%\n"
                            f"Readability ({formula_name}): {readability['flesch_score']} — {readability['flesch_level']}"
                        ))
                    else:
                        self.readability_label.config(text="")
                    self._set_busy(False)
                elif kind == "error":
                    self.status_label.config(text=f"Error: {item[1]}")
                    self.progress.stop()
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self.poll_queue)

    def _tick_elapsed(self):
        # NOT self.stop_btn.cget("state") == "normal": ttk's cget("state")
        # returns a _tkinter.Tcl_Obj, not a plain str — confirmed directly
        # that comparing it to a string literal is always False regardless
        # of the button's actual state, which silently froze this elapsed
        # timer at 00:00 for the whole run. .state() (used the same way
        # elsewhere in this file) returns real state-flag strings instead.
        if self.run_start_time is not None and "disabled" not in self.stop_btn.state():
            elapsed = int(time.monotonic() - self.run_start_time)
            self.elapsed_label.config(text=f"⏱ {elapsed // 60:02d}:{elapsed % 60:02d}")
        self.after(500, self._tick_elapsed)

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.run_btn.config(state="disabled" if busy or not self.filepath else "normal")
        self.select_btn.config(state=state)
        self._refresh_export_buttons(busy)
        # Gated on book_sentences, not glossary_entries: free-text search
        # doesn't need a single dictionary match to be useful, so it's
        # available even for an undefined/unsupported-language book.
        self.concordance_btn.config(state="disabled" if busy or not self.book_sentences else "normal")
        can_study = not busy and bool(self.glossary_entries)
        self.study_flashcard_btn.config(state="normal" if can_study else "disabled")
        self.study_writing_btn.config(state="normal" if can_study else "disabled")
        self.study_anki_btn.config(state="normal" if can_study else "disabled")
        not_known = [w for w, _ in self.glossary_entries if w not in self.known_words] if self.glossary_entries else []
        ready = len(not_known)
        due = len(due_words(self.progress_schedule, not_known)) if not_known else 0
        text = f"{ready} word(s) ready to study"
        # due == ready the very first time a word is ever seen (an
        # unscheduled word counts as due) — the distinction only becomes
        # informative once some of this book's words have been reviewed
        # before, so it's only worth a second line then.
        if 0 < due < ready:
            text += f"\n{due} due for review today"
        self.study_count_label.config(text=text)
        self._refresh_reading_readiness()
        self._refresh_progress_stat()

        can_quiz = not busy and bool(self.grammar_results)
        self.grammar_quiz_btn.config(state="normal" if can_quiz else "disabled")
        self.grammar_identify_btn.config(state="normal" if can_quiz else "disabled")
        self.grammar_anki_btn.config(state="normal" if can_quiz else "disabled")
        self._refresh_grammar_readiness()
        self._refresh_grammar_progress_stat()

        self.stop_btn.config(state="normal" if busy else "disabled")
        if busy:
            self.cancel_event.clear()
        else:
            self.progress.stop()
            self.run_start_time = None

    def _refresh_export_buttons(self, busy=False):
        # save_btn/anki_btn in the footer are SHARED across both
        # sections (see save_as()/export_anki()'s own active_section
        # branch) — their enabled state has to follow whichever
        # section's data actually exists right now, not always
        # glossary_entries. The docked cards' own buttons don't need
        # this: each only exists inside its own section's frame, so it
        # never needs to reflect the OTHER section's data.
        if self.active_section == "vocab":
            has_data = bool(self.glossary_entries)
        elif self.active_section == "grammar":
            has_data = bool(self.grammar_results)
        else:
            has_data = bool(self.phrases)
        self.save_btn.config(state="disabled" if busy or not has_data else "normal")
        self.anki_btn.config(state="disabled" if busy or not has_data else "normal")

    # ------------------------------------------------------- pronunciation

    def speak(self, word):
        if not word:
            return
        lang = self.detected_lang or "en"
        self.status_label.config(text=f"Speaking “{word}”...")
        threading.Thread(target=self._run_speak, args=(word, lang), daemon=True).start()

    def _run_speak(self, word, lang):
        try:
            speak_word(word, lang)
            self.work_queue.put(("status", "Ready."))
        except Exception as exc:
            self.work_queue.put(("status", f"Couldn't pronounce “{word}” (needs internet): {exc}"))

    # -------------------------------------------------------------- render

    def toggle_sort(self):
        self.sort_by_freq = not self.sort_by_freq
        self.sort_btn.config(text="SORT: FREQUENCY" if self.sort_by_freq else "SORT: A-Z")
        self.render_results()

    def _sorted_entries(self):
        return exp.sorted_entries(self.glossary_entries, self.word_freqs, self.sort_by_freq)

    def _build_families(self):
        """Group this glossary's words by shared root ("book"/"books"/
        "booked" all under "book") using word_family_root() — English
        only, since that relies on WordNet's morphy analyzer (see its
        docstring, and word_family_root()'s own, for why Spanish has no
        equivalent bundled here). Runs once per glossary, not per
        render, so toggling the grouping checkbox or typing in FILTER
        stays instant.

        The "Level" filter is a SEPARATE gate from grouping — English
        and Spanish both have real CEFR data now (cefr_en.json/
        cefr_es.json, see build_cefr_map()'s docstring), so it's enabled
        for either, even though grouping itself stays English-only.
        """
        self.family_map = {}
        self.word_family_root_of = {}
        self.family_definition = {}
        self.family_pos = {}
        self.family_cefr = {}
        is_english = self.detected_lang == "en"
        has_cefr_data = self.detected_lang in ("en", "es")
        self.group_families_check.config(state="normal" if is_english else "disabled")
        self.level_filter_combo.config(state="readonly" if has_cefr_data else "disabled")
        if not has_cefr_data:
            self.level_filter_var.set("All levels")
        if not is_english:
            self.group_families_var.set(False)
            return
        groups = {}
        for word, _definition in self.glossary_entries:
            key = word_family_root(word, "eng") or word
            groups.setdefault(key, []).append(word)
        for key, words in groups.items():
            if len(words) < 2:
                continue
            self.family_map[key] = sorted(words)
            for w in words:
                self.word_family_root_of[w] = key
            self.family_definition[key] = define(key, "eng")
            self.family_pos[key] = word_pos(key, "eng")
            self.family_cefr[key] = cefr_level(key, self.family_pos[key])

    def _grouped_display(self):
        """Entries from _sorted_entries(), with family members collapsed
        into a single ("family", root, [(word, definition), ...]) item at
        the position of their first (sort-order) member, and every other
        word passed through as ("single", word, definition).
        """
        entries = self._sorted_entries()
        def_by_word = dict(entries)
        seen_roots = set()
        display = []
        for word, definition in entries:
            root = self.word_family_root_of.get(word)
            if root:
                if root in seen_roots:
                    continue
                seen_roots.add(root)
                members = [(w, def_by_word.get(w, "")) for w in self.family_map[root]]
                display.append(("family", root, members))
            else:
                display.append(("single", word, definition))
        return display

    def render_results(self):
        # Captured as the full multi-selection, not just self.selected_word
        # -- marking a whole batch known/learning re-renders the list (to
        # update icons/hide known rows), and without re-selecting
        # everything that's still visible afterward, a second bulk action
        # on the same batch would silently do nothing (confirmed with the
        # single-word case earlier; the same gap applies to a multi-select).
        previous_selection = list(self.tree.selection())
        query = self.search_var.get().strip().lower()
        hide_known = self.hide_known_var.get()
        level_filter = self.level_filter_var.get()
        group_mode = self.group_families_var.get() and bool(self.family_map)
        self.tree.delete(*self.tree.get_children())
        self.word_lookup = {}
        shown = 0

        def word_matches(word):
            if query and not word.startswith(query):
                return False
            if hide_known and word in self.known_words:
                return False
            if level_filter and level_filter != "All levels":
                level = self.word_cefr_map.get(word)
                if level_filter == "No level":
                    if level is not None:
                        return False
                elif level != level_filter:
                    return False
            return True

        def insert_word_row(parent, word, definition):
            count = self.word_freqs.get(word)
            example = self.word_examples.get(word)
            pos = self.word_pos_map.get(word)
            level = self.word_cefr_map.get(word)
            self.word_lookup[word] = (definition, count, example, pos, level)
            status = "known" if word in self.known_words else ("learning" if word in self.learning_words else "new")
            short_def = definition if len(definition) <= 90 else definition[:87] + "..."
            self.tree.insert(
                parent, "end", iid=word, text=f"{STATUS_ICON[status]} {word}".strip(),
                values=(count or "", pos or "", level or "", short_def), tags=(status,),
            )

        if group_mode:
            for kind, *rest in self._grouped_display():
                if kind == "single":
                    word, definition = rest
                    if not word_matches(word):
                        continue
                    insert_word_row("", word, definition)
                    shown += 1
                else:
                    root, members = rest
                    visible = [(w, d) for w, d in members if word_matches(w)]
                    if not visible:
                        continue
                    total_count = sum(self.word_freqs.get(w, 0) for w, _d in visible)
                    root_def = self.family_definition.get(root) or ""
                    short_root_def = root_def if len(root_def) <= 90 else root_def[:87] + "..."
                    root_pos = self.family_pos.get(root) or ""
                    root_cefr = self.family_cefr.get(root) or ""
                    family_iid = f"family::{root}"
                    self.tree.insert(
                        "", "end", iid=family_iid, text=f"▸ {root}  ({len(visible)} forms)",
                        values=(total_count or "", root_pos, root_cefr, short_root_def), tags=("family",),
                        open=bool(query),
                    )
                    for word, definition in visible:
                        insert_word_row(family_iid, word, definition)
                        shown += 1
        else:
            for word, definition in self._sorted_entries():
                if not word_matches(word):
                    continue
                insert_word_row("", word, definition)
                shown += 1
        if not shown and self.glossary_entries and not query:
            if level_filter != "All levels":
                message = "no matching words"
            else:
                message = "(all matching words are hidden — see \"Hide words I already know\")"
            self.tree.insert("", "end", text=message, values=("", "", "", ""))
        elif not self.glossary_entries and self.undefined_words and not query:
            self.tree.insert(
                "", "end", text="No offline dictionary matched any word here",
                values=("", "", "", "This book's words are still correctly extracted — try a different BOOK LANGUAGE, or export the word list as-is."),
            )
        still_visible = [w for w in previous_selection if w in self.word_lookup]
        if still_visible:
            self.tree.selection_set(still_visible)
            self.tree.see(still_visible[0])
            self._show_selection(still_visible)
        else:
            self._clear_detail_panel()

    def _on_select_word(self, _event):
        sel = [w for w in self.tree.selection() if w in self.word_lookup]
        if not sel:
            self._clear_detail_panel()
            return
        self._show_selection(sel)

    def _show_selection(self, words):
        """Populate the Detail panel for the current Treeview selection —
        `words` is 1 or more word ids. A single selection shows its full
        definition/example; a multi-selection (selectmode="extended" lets
        the reader Ctrl/Shift-click or drag across many rows) shows a
        count and enables only the action that makes sense in bulk
        (Know/Learning) — Speak and Find All stay single-word-only.
        """
        self.selected_words = words
        self.selected_word = words[0] if len(words) == 1 else None
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        if len(words) == 1:
            word = words[0]
            definition, count, example, pos, level = self.word_lookup[word]
            label = word
            if count:
                label += f"  ({count}×)"
            if pos:
                label += f"  · {pos}"
            if level:
                label += f"  · {level}"
            self.detail_word_var.set(label)
            self.detail_text.insert("end", definition, "def")
            if example:
                self.detail_text.insert("end", f"\n\n“{example}”", "example")
            self.detail_speak_btn.config(state="normal")
            self.detail_find_btn.config(state="normal" if self.book_sentences else "disabled")
            self.detail_word_info_btn.config(state="normal")
            self.detail_focus_btn.config(state="disabled")
        else:
            self.detail_word_var.set(f"{len(words)} words selected")
            self.detail_text.insert(
                "end", "Mark all of them known/learning below, or press Ctrl+C to copy them, or open "
                "FOCUS VIEW to review them full-screen.", "hint",
            )
            self.detail_speak_btn.config(state="disabled")
            self.detail_find_btn.config(state="disabled")
            self.detail_word_info_btn.config(state="disabled")
            self.detail_focus_btn.config(state="normal")
        self.detail_text.config(state="disabled")
        self.detail_know_btn.config(state="normal")
        self.detail_learning_btn.config(state="normal")

    def _clear_detail_panel(self):
        self.selected_word = None
        self.selected_words = []
        self.detail_word_var.set("Select a word to see details")
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.config(state="disabled")
        for btn in (
            self.detail_speak_btn, self.detail_find_btn, self.detail_word_info_btn, self.detail_focus_btn,
            self.detail_know_btn, self.detail_learning_btn,
        ):
            btn.config(state="disabled")

    # ------------------------------------------------------- grammar section

    def _render_grammar_results(self):
        """(Re)builds the Grammar tree from self.grammar_results — called
        once per pipeline run (there's no filter box to re-trigger it,
        unlike render_results()). Two-level rows, one per category (in
        CATEGORY_ORDER) auto-expanded, with matched rules nested
        underneath — the same nested-row mechanism the vocabulary tree
        already uses for word-family grouping, reused rather than
        inventing a second grouping widget.
        """
        self.grammar_tree.delete(*self.grammar_tree.get_children())
        self._clear_grammar_detail_panel()
        if not self.grammar_results:
            return
        by_category = {}
        for rule_id, data in self.grammar_results.items():
            by_category.setdefault(data["category"], []).append((rule_id, data))
        category_order = CATEGORY_ORDER_ES if self.detected_lang == "es" else CATEGORY_ORDER
        for category in category_order:
            rules = by_category.get(category)
            if not rules:
                continue
            cat_iid = f"category::{category}"
            self.grammar_tree.insert("", "end", iid=cat_iid, text=category, values=("",), tags=("category",), open=True)
            for rule_id, data in sorted(rules, key=lambda item: item[1]["name"]):
                status = (
                    "known" if rule_id in self.grammar_known
                    else "learning" if rule_id in self.grammar_learning else "new"
                )
                self.grammar_tree.insert(
                    cat_iid, "end", iid=rule_id, text=f"{STATUS_ICON[status]} {data['name']}".strip(),
                    values=(data["total_count"],), tags=(status,),
                )

    def _on_select_grammar_rule(self, item_id):
        # Category header rows aren't in self.grammar_results (only rule
        # rows are, keyed by the same id used as their tree iid) — a
        # header click naturally falls through to the idle state below,
        # since selecting them isn't a meaningful action.
        if item_id and item_id in self.grammar_results:
            self._show_grammar_rule(item_id)
        else:
            self._clear_grammar_detail_panel()

    def _show_grammar_rule(self, rule_id):
        self.selected_grammar_rule = rule_id
        data = self.grammar_results[rule_id]
        header = f"{data['name']}  ({data['total_count']} sentence{'s' if data['total_count'] != 1 else ''})"
        if rule_id in self.grammar_known:
            header += "  · known"
        elif rule_id in self.grammar_learning:
            header += "  · learning"
        self.grammar_detail_header_var.set(header)
        text = self.grammar_detail_text
        text.config(state="normal")
        text.delete("1.0", "end")
        text.insert("end", data["definition"], "def")
        text.insert("end", "\n\n" + data["rule_explanation"] + "\n\n", "rule")
        sentences = data["sentences"]
        total = data["total_count"]
        if total > len(sentences):
            text.insert("end", f"Showing the first {len(sentences)} of {total} sentences:\n\n", "hint")
        for i, sentence in enumerate(sentences, start=1):
            text.insert("end", f"{i}.  ", "num")
            text.insert("end", sentence + "\n\n", "example")
        text.config(state="disabled")
        self.grammar_know_btn.config(state="normal")
        self.grammar_learning_btn.config(state="normal")

    def _clear_grammar_detail_panel(self):
        self.selected_grammar_rule = None
        self.grammar_know_btn.config(state="disabled")
        self.grammar_learning_btn.config(state="disabled")
        text = self.grammar_detail_text
        text.config(state="normal")
        text.delete("1.0", "end")
        if self.detected_lang is None:
            self.grammar_detail_header_var.set("Select a grammar structure to see details")
        elif self.detected_lang not in ("en", "es"):
            self.grammar_detail_header_var.set("Grammar analysis isn't available for this book's language")
            text.insert(
                "end",
                f"Grammar analysis is only available for English and Spanish books right now. This book was "
                f"detected as {language_name(self.detected_lang)}. Detecting tenses, moods, and so on relies on "
                "language-specific rules with no equivalent yet built for other languages.",
                "hint",
            )
        elif not self.grammar_results:
            self.grammar_detail_header_var.set("No grammar structures were detected")
            text.insert("end", "No grammar structures were detected in this book.", "hint")
        else:
            self.grammar_detail_header_var.set("Select a grammar structure to see details")
        text.config(state="disabled")

    def _phrases_are_empty(self):
        return not self.phrases.get("recognized") and not self.phrases.get("recurring")

    def _render_phrases(self):
        """(Re)builds the Phrases tree from self.phrases — triggered both
        by a fresh pipeline run and by every keystroke in the SEARCH box
        (self.phrase_search_var's trace), same live-filter behavior as
        Vocabulary's own FILTER box. Unlike Grammar, this has no
        per-language "not available" state to show: extract_phrases()'s
        "recurring" tier works for every language this app supports (see
        its own docstring), so an empty result here really does mean "no
        book loaded" or "nothing found," not a language gap — only the
        "recognized" tier is English/Spanish-only, silently absent
        otherwise.
        """
        self.phrase_tree.delete(*self.phrase_tree.get_children())
        self._clear_phrase_detail_panel()
        if self._phrases_are_empty():
            return
        query = self.phrase_search_var.get().strip().lower()
        populate_phrase_tree(self.phrase_tree, self.phrases, query)

    def _on_select_phrase(self, item_id):
        entry = self._phrase_entry_by_iid(item_id) if item_id else None
        if entry is None:
            self._clear_phrase_detail_panel()
            return
        self._show_phrase(entry)

    def _show_phrase(self, entry):
        self.selected_phrase = entry["phrase"]
        is_recognized = "idiomatic" in entry
        header = f"{entry['phrase']}  ({entry['count']} occurrence{'s' if entry['count'] != 1 else ''})"
        if is_recognized:
            header += "  · Idiom" if entry["idiomatic"] else f"  · {entry['pos']}"
        self.phrase_detail_header_var.set(header)
        text = self.phrase_detail_text
        text.config(state="normal")
        text.delete("1.0", "end")
        if is_recognized:
            text.insert("end", entry["definition"] + "\n\n", "def")
            text.insert(
                "end",
                ("A real idiom, per Wiktionary's own editorial tagging" if entry["idiomatic"] else
                 "A real, dictionary-catalogued phrase (Wiktionary), though not specifically tagged "
                 "as a figurative idiom") + " — not a statistical guess.\n\n",
                "stat",
            )
        elif entry.get("meaning"):
            text.insert("end", entry["meaning"] + "\n\n", "def")
        else:
            text.insert(
                "end",
                "No dictionary meaning available — this word combination isn't independently catalogued "
                "as its own phrase anywhere (see extract_phrases()'s docstring: most recurring sequences, "
                "however real, simply aren't independently lexicalized anywhere).\n\n",
                "hint",
            )
        if not is_recognized and entry["pmi"] is not None:
            text.insert(
                "end",
                f"PMI (statistical stickiness): {entry['pmi']} — higher means these words co-occur far more "
                "than their individual frequencies alone would predict.\n\n",
                "stat",
            )
        matches, total = find_occurrences(self.book_sentences, entry["phrase"], whole_word=True, max_results=300)
        if total > len(matches):
            text.insert("end", f"Showing the first {len(matches)} of {total} occurrences:\n\n", "stat")
        for i, (sentence, spans) in enumerate(matches, start=1):
            text.insert("end", f"{i}.  ", "num")
            pos = 0
            for start, end in spans:
                text.insert("end", sentence[pos:start], "example")
                text.insert("end", sentence[start:end], "hint")
                pos = end
            text.insert("end", sentence[pos:] + "\n\n", "example")
        text.config(state="disabled")

    def _clear_phrase_detail_panel(self):
        self.selected_phrase = None
        text = self.phrase_detail_text
        text.config(state="normal")
        text.delete("1.0", "end")
        if self._phrases_are_empty():
            self.phrase_detail_header_var.set(
                "Select a phrase to see where it's used" if self.book_sentences else "No book loaded yet",
            )
            if self.book_sentences:
                text.insert("end", "No recognized or recurring phrases were found in this book.", "hint")
        else:
            self.phrase_detail_header_var.set("Select a phrase to see where it's used")
        text.config(state="disabled")

    def _grammar_mark(self, status):
        rule_id = self.selected_grammar_rule
        if not rule_id:
            return
        if status == "known":
            self.grammar_known.add(rule_id)
            self.grammar_learning.discard(rule_id)
        else:
            self.grammar_learning.add(rule_id)
            self.grammar_known.discard(rule_id)
        # Same "marking here still counts as a real SM-2 review" reasoning
        # as _detail_mark()'s vocab equivalent — otherwise a structure
        # marked known straight from the tree would stay "due" and get
        # shown again immediately in the very next Grammar Quiz session.
        review_word(self.grammar_schedule, rule_id, status == "known")
        if status == "known":
            mark_known(self.grammar_learned_at, rule_id)
        self._save_progress()
        self._refresh_grammar_stat()
        self._refresh_grammar_readiness()
        self._refresh_grammar_progress_stat()
        self._render_grammar_results()
        # _render_grammar_results() rebuilds the tree from scratch and
        # clears the detail panel — re-select and re-show explicitly
        # (not just rely on selection_set() re-firing <<TreeviewSelect>>,
        # which ttk doesn't reliably do when nothing about the selection
        # itself changed) so the reader immediately sees the updated
        # know/learning status on the very row they just marked.
        if self.grammar_tree.exists(rule_id):
            self.grammar_tree.selection_set(rule_id)
            self.grammar_tree.see(rule_id)
            self._show_grammar_rule(rule_id)

    # ---------------------------------------------- grammar tree copy/menu

    def _build_grammar_tree_context_menu(self):
        p = self.palette
        self.grammar_tree_menu = tk.Menu(
            self, tearoff=0, bg=p["PANEL"], fg=p["FG"], activebackground=p["SECOND_HOVER"],
            activeforeground=p["FG"], font=th.FONT,
        )
        self.grammar_tree_menu.add_command(label="✅ Mark as known", command=lambda: self._grammar_mark("known"))
        self.grammar_tree_menu.add_command(
            label="📖 Mark as still learning", command=lambda: self._grammar_mark("learning"),
        )
        self.grammar_tree_menu.add_separator()
        self.grammar_tree_menu.add_command(label="Copy structure name", command=self._copy_grammar_names_only)
        self.grammar_tree_menu.add_command(label="Copy definition", command=self._copy_grammar_definitions_only)
        self.grammar_tree_menu.add_command(label="Copy all sentences", command=self._copy_grammar_sentences_only)
        self.grammar_tree_menu.add_command(label="Copy (Ctrl+C)", command=self._copy_grammar_selection)
        self.themed.add(
            self.grammar_tree_menu, bg="PANEL", fg="FG", activebackground="SECOND_HOVER", activeforeground="FG",
        )

    def _show_grammar_tree_context_menu(self, event):
        row = self.grammar_tree.identify_row(event.y)
        if row and row not in self.grammar_tree.selection():
            self.grammar_tree.selection_set(row)
        sel = [r for r in self.grammar_tree.selection() if r in self.grammar_results]
        if not sel:
            return
        self.grammar_tree_menu.tk_popup(event.x_root, event.y_root)

    def _copy_grammar_selection(self, _event=None):
        rule_ids = [r for r in self.grammar_tree.selection() if r in self.grammar_results]
        if not rule_ids:
            return "break"
        lines = []
        for rule_id in rule_ids:
            data = self.grammar_results[rule_id]
            lines.append(f"{data['name']}\t{data['total_count']}\t{data['definition']}\t{data['rule_explanation']}")
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        plural = "s" if len(rule_ids) != 1 else ""
        self.status_label.config(text=f"Copied {len(rule_ids)} structure{plural} to the clipboard.")
        return "break"

    def _copy_grammar_names_only(self):
        rule_ids = [r for r in self.grammar_tree.selection() if r in self.grammar_results]
        if not rule_ids:
            return
        names = [self.grammar_results[r]["name"] for r in rule_ids]
        self.clipboard_clear()
        self.clipboard_append("\n".join(names))
        plural = "s" if len(rule_ids) != 1 else ""
        self.status_label.config(text=f"Copied {len(rule_ids)} structure name{plural} to the clipboard.")

    def _copy_grammar_definitions_only(self):
        rule_ids = [r for r in self.grammar_tree.selection() if r in self.grammar_results]
        if not rule_ids:
            return
        definitions = [self.grammar_results[r]["definition"] for r in rule_ids]
        self.clipboard_clear()
        self.clipboard_append("\n".join(definitions))
        plural = "s" if len(rule_ids) != 1 else ""
        self.status_label.config(text=f"Copied {len(rule_ids)} definition{plural} to the clipboard.")

    def _copy_grammar_sentences_only(self):
        rule_ids = [r for r in self.grammar_tree.selection() if r in self.grammar_results]
        if not rule_ids:
            return
        lines = []
        for rule_id in rule_ids:
            lines.extend(self.grammar_results[rule_id]["sentences"])
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self.status_label.config(text=f"Copied {len(lines)} sentence(s) to the clipboard.")

    # -------------------------------------------------------- phrase tree

    def _phrase_entry_by_iid(self, iid):
        """The entry dict for a phrase tree row iid, or None for a
        section/category header or stale iid — see
        phrase_analyzer.populate_phrase_tree()'s docstring for the two
        iid shapes this parses: "recognized::<phrase>" (a
        {"phrase","pos","definition","idiomatic","count","example"}
        entry from the flat "recognized" list) or
        "recurring::<length>::<phrase>" (a {"phrase","count","pmi",
        "example","meaning"} entry from "recurring"'s per-length
        lists). Factored out here since every copy action below needs
        it too, not just selection.
        """
        if iid.startswith("recognized::"):
            phrase = iid[len("recognized::"):]
            return next((e for e in self.phrases.get("recognized", []) if e["phrase"] == phrase), None)
        if iid.startswith("recurring::"):
            _prefix, length_str, phrase = iid.split("::", 2)
            try:
                length = int(length_str)
            except ValueError:
                return None
            entries = self.phrases.get("recurring", {}).get(length, {}).get("entries", [])
            return next((e for e in entries if e["phrase"] == phrase), None)
        return None

    def _build_phrase_tree_context_menu(self):
        p = self.palette
        self.phrase_tree_menu = tk.Menu(
            self, tearoff=0, bg=p["PANEL"], fg=p["FG"], activebackground=p["SECOND_HOVER"],
            activeforeground=p["FG"], font=th.FONT,
        )
        self.phrase_tree_menu.add_command(label="Copy phrase", command=self._copy_phrase_names_only)
        self.phrase_tree_menu.add_command(label="Copy meaning", command=self._copy_phrase_meanings_only)
        self.phrase_tree_menu.add_command(label="Copy all sentences", command=self._copy_phrase_sentences_only)
        self.phrase_tree_menu.add_command(label="Copy (Ctrl+C)", command=self._copy_phrase_selection)
        self.themed.add(
            self.phrase_tree_menu, bg="PANEL", fg="FG", activebackground="SECOND_HOVER", activeforeground="FG",
        )

    def _show_phrase_tree_context_menu(self, event):
        row = self.phrase_tree.identify_row(event.y)
        if row and row not in self.phrase_tree.selection():
            self.phrase_tree.selection_set(row)
        sel = [r for r in self.phrase_tree.selection() if self._phrase_entry_by_iid(r)]
        if not sel:
            return
        self.phrase_tree_menu.tk_popup(event.x_root, event.y_root)

    def _selected_phrase_entries(self):
        entries = [self._phrase_entry_by_iid(r) for r in self.phrase_tree.selection()]
        return [e for e in entries if e]

    def _copy_phrase_selection(self, _event=None):
        entries = self._selected_phrase_entries()
        if not entries:
            return "break"
        lines = []
        for e in entries:
            # "recognized" entries have "definition"/"idiomatic"; "recurring"
            # ones have "pmi"/"meaning" instead — see _phrase_entry_by_iid()'s
            # docstring for the two shapes. .get() throughout so a field
            # missing from whichever shape this entry actually is just
            # comes out blank instead of raising.
            info = e["definition"] if "idiomatic" in e else (e.get("meaning") or "")
            lines.append(f"{e['phrase']}\t{e['count']}\t{e.get('pmi', '') or ''}\t{info}")
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        plural = "s" if len(entries) != 1 else ""
        self.status_label.config(text=f"Copied {len(entries)} phrase{plural} to the clipboard.")
        return "break"

    def _copy_phrase_names_only(self):
        entries = self._selected_phrase_entries()
        if not entries:
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(e["phrase"] for e in entries))
        plural = "s" if len(entries) != 1 else ""
        self.status_label.config(text=f"Copied {len(entries)} phrase{plural} to the clipboard.")

    def _copy_phrase_meanings_only(self):
        # "recognized" entries always have a real "definition"; "recurring"
        # ones only sometimes have a WordNet "meaning" — see
        # _phrase_entry_by_iid()'s docstring for the two shapes.
        meanings = [
            (e["definition"] if "idiomatic" in e else e.get("meaning"))
            for e in self._selected_phrase_entries()
        ]
        meanings = [m for m in meanings if m]
        if not meanings:
            self.status_label.config(text="No dictionary meaning available for the selected phrase(s).")
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(meanings))
        plural = "s" if len(meanings) != 1 else ""
        self.status_label.config(text=f"Copied {len(meanings)} meaning{plural} to the clipboard.")

    def _copy_phrase_sentences_only(self):
        entries = self._selected_phrase_entries()
        if not entries:
            return
        lines = []
        for entry in entries:
            matches, _total = find_occurrences(self.book_sentences, entry["phrase"], whole_word=True, max_results=300)
            lines.extend(sentence for sentence, _spans in matches)
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self.status_label.config(text=f"Copied {len(lines)} sentence(s) to the clipboard.")

    # ------------------------------------------------------- grammar stats

    def _refresh_grammar_stat(self):
        self.grammar_header_label.config(text="YOUR GRAMMAR")
        known, learning = len(self.grammar_known), len(self.grammar_learning)
        if known or learning:
            text = f"{known} known\n{learning} still learning\n(across all books)"
        else:
            text = "None yet — use the Grammar Quiz to start."
        self.grammar_stat_label.config(text=text)

    def _refresh_grammar_readiness(self):
        if not self.grammar_results:
            idle = "No grammar structures detected." if self.detected_lang is not None else "No book loaded yet."
            self.grammar_readiness_book_label.config(text=idle)
            self.grammar_readiness_meta_label.config(text="")
            self.grammar_readiness_label.config(text="")
            self.grammar_readiness_bar.config(value=0)
            self.grammar_readiness_known_label.config(text="")
            self.grammar_readiness_to_learn_label.config(text="")
            self.grammar_quiz_count_label.config(text="0 structure(s) ready to study")
            return

        title = "?"
        if self.filepath:
            title = os.path.splitext(os.path.basename(self.filepath))[0]
            if len(title) > 34:
                title = title[:31] + "..."
        self.grammar_readiness_book_label.config(text=title)

        total = len(self.grammar_results)
        self.grammar_readiness_meta_label.config(text=f"{total} structure(s) found")

        known_in_book = sum(1 for rid in self.grammar_results if rid in self.grammar_known)
        to_learn = total - known_in_book
        pct = round(known_in_book / total * 100) if total else 0
        self.grammar_readiness_label.config(text=f"📊 Grammar readiness: {pct}%")
        self.grammar_readiness_bar.config(value=pct)
        self.grammar_readiness_known_label.config(text=f"✅ Known: {known_in_book:,} structures")
        self.grammar_readiness_to_learn_label.config(text=f"📖 To learn: {to_learn:,} structures")

        not_known = [rid for rid in self.grammar_results if rid not in self.grammar_known]
        ready = len(not_known)
        due = len(due_words(self.grammar_schedule, not_known)) if not_known else 0
        text = f"{ready} structure(s) ready to study"
        if 0 < due < ready:
            text += f"\n{due} due for review today"
        self.grammar_quiz_count_label.config(text=text)

    def _refresh_grammar_progress_stat(self):
        this_week = words_learned_since(self.grammar_learned_at, 7)
        this_month = words_learned_since(self.grammar_learned_at, 30)
        lines = [
            f"📅 This week: {this_week} structure(s) learned",
            f"📅 This month: {this_month} structure(s) learned",
            f"📅 All time: {len(self.grammar_known):,} structure(s) known",
        ]
        self.grammar_progress_stat_label.config(text="\n".join(lines))
        self._refresh_grammar_restore_button()

    def _refresh_grammar_restore_button(self):
        # "grammar" is a fixed sentinel key into the SAME self.reset_backups
        # dict the per-language vocab backups use — never a real ISO
        # 639-1 code, so it can't collide with one.
        backup = restorable_backup(self.reset_backups, "grammar")
        self.grammar_restore_btn.config(state="normal" if backup else "disabled")

    def _draw_grammar_chart(self):
        c = self.grammar_chart_canvas
        c.delete("all")
        if not self.grammar_results:
            return
        category_order = CATEGORY_ORDER_ES if self.detected_lang == "es" else CATEGORY_ORDER
        counts = {cat: 0 for cat in category_order}
        for data in self.grammar_results.values():
            counts[data["category"]] += data["total_count"]
        buckets = [(cat, counts[cat]) for cat in category_order if counts[cat] > 0]
        if not buckets:
            return
        max_val = max(v for _, v in buckets) or 1
        p = self.palette
        width = max(int(c.winfo_width()) or 220, 220)
        # Label on its own line, directly above its bar — not beside it.
        # A side-by-side layout (like the word-frequency chart's fixed
        # "1-10"/"11-50" bucket labels) needs the label and bar to each
        # fit in a fixed share of the width; several of these category
        # names ("Clauses & Conditionals", "Non-finite Forms") were
        # actually wider than that fixed share at this app's font/DPI,
        # so the bar was drawn right on top of the tail end of the text
        # instead of just being packed a bit tight. Stacking removes the
        # competition entirely: the label gets the FULL width on its own
        # row, so no category name can ever collide with its bar again.
        row_h = 34
        for i, (label, value) in enumerate(buckets):
            y = i * row_h + 6
            c.create_text(4, y, anchor="nw", text=label, fill=p["FG"], font=th.FONT_SMALL_BOLD)
            bar_y = y + 16
            bar_max = max(width - 50, 40)
            bar_w = int(bar_max * value / max_val)
            c.create_rectangle(4, bar_y, 4 + bar_w, bar_y + 12, fill=p["ACCENT"], width=0)
            c.create_text(10 + bar_w, bar_y + 6, anchor="w", text=str(value), fill=p["DIM"], font=th.FONT_SMALL_BOLD)
        c.config(height=len(buckets) * row_h + 12)

    # -------------------------------------------------------- grammar quiz

    def open_grammar_quiz(self, mode="structure"):
        if not self.grammar_results:
            return
        if mode == "identify":
            # A sentence card is worth asking only while at least one of
            # its matched rules isn't already fully known — see
            # GrammarQuizWindow.show_card()'s own per-card skip check for
            # the same rule, applied here just to decide whether it's
            # worth opening the window at all.
            index = build_sentence_index(self.grammar_results)
            cards = [
                (sentence, matches) for sentence, matches in index.items()
                if not all(rid in self.grammar_known for rid, _d in matches)
            ]
            empty_message = (
                "Nothing left to identify here — every sentence's grammar is already marked known."
            )
        else:
            cards = [
                (rid, data) for rid, data in self.grammar_results.items() if rid not in self.grammar_known
            ]
            empty_message = "Nothing left to study here — every detected structure is already marked known."
        if cards:
            GrammarQuizWindow(
                self, cards, on_finish=self._record_grammar_quiz_results, palette=self.palette,
                schedule=self.grammar_schedule, already_known_rules=self.grammar_known,
                learned_at=self.grammar_learned_at, mode=mode,
            )
        else:
            messagebox.showinfo("Grammar Quiz", empty_message)

    def _record_grammar_quiz_results(self, known_rule_ids, learning_rule_ids):
        # schedule/learned_at were already updated in place during the
        # session (GrammarQuizWindow calls review_word()/mark_known() per
        # card) — same reasoning as _record_study_results()'s vocab
        # equivalent, so only known/learning need merging here.
        self.grammar_known |= set(known_rule_ids)
        self.grammar_learning |= set(learning_rule_ids)
        self.grammar_learning -= self.grammar_known
        self._save_progress()
        self._refresh_grammar_stat()
        self._refresh_grammar_readiness()
        self._refresh_grammar_progress_stat()
        self._render_grammar_results()

    def reset_grammar_progress(self):
        if not self.grammar_known and not self.grammar_learning:
            return
        if messagebox.askyesno(
            "Reset grammar progress",
            "Clear your grammar structure history (known + still-learning)? Vocabulary progress is untouched."
            f"\n\nYou can undo this from RESTORE GRAMMAR for up to {RESET_BACKUP_DAYS} days.",
        ):
            backup_before_reset(
                self.reset_backups, "grammar",
                self.grammar_known, self.grammar_learning, self.grammar_schedule, self.grammar_learned_at,
            )
            self.grammar_known.clear()
            self.grammar_learning.clear()
            self.grammar_schedule.clear()
            self.grammar_learned_at.clear()
            self._save_progress()
            self._refresh_grammar_stat()
            self._refresh_grammar_readiness()
            self._refresh_grammar_progress_stat()
            self._render_grammar_results()

    def restore_grammar_progress(self):
        backup = restorable_backup(self.reset_backups, "grammar")
        if not backup:
            return
        if messagebox.askyesno(
            "Restore grammar progress",
            f"Restore your grammar progress from just before the last reset "
            f"({len(backup['known'])} known, {len(backup['learning'])} still learning)?",
        ):
            self.grammar_known.clear()
            self.grammar_known.update(backup["known"])
            self.grammar_learning.clear()
            self.grammar_learning.update(backup["learning"])
            self.grammar_schedule.clear()
            self.grammar_schedule.update(backup["schedule"])
            self.grammar_learned_at.clear()
            self.grammar_learned_at.update(backup["learned_at"])
            del self.reset_backups["grammar"]
            self._save_progress()
            self._refresh_grammar_stat()
            self._refresh_grammar_readiness()
            self._refresh_grammar_progress_stat()
            self._render_grammar_results()

    # --------------------------------------------------- grammar exporting

    def export_grammar_anki(self):
        if not self.grammar_results:
            return
        deck_name = "Grammar"
        if self.filepath:
            deck_name = os.path.splitext(os.path.basename(self.filepath))[0] + "_Grammar"
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Anki-importable text file", "*.txt")],
            initialfile=f"{deck_name}_anki.txt",
        )
        if not path:
            return
        try:
            exp.export_grammar_anki(path, self.grammar_results, self.grammar_known, self.grammar_learning, deck_name)
        except Exception as exc:
            self.status_label.config(text=f"Error exporting to Anki: {exc}")
            return
        self.status_label.config(text=f"Saved Anki import file to {path} — open it from Anki's File > Import.")

    def _export_phrases_anki(self):
        if not self.phrases:
            return
        deck_name = "Phrases"
        if self.filepath:
            deck_name = os.path.splitext(os.path.basename(self.filepath))[0] + "_Phrases"
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Anki-importable text file", "*.txt")],
            initialfile=f"{deck_name}_anki.txt",
        )
        if not path:
            return
        try:
            exp.export_phrases_anki(path, self.phrases, deck_name)
        except Exception as exc:
            self.status_label.config(text=f"Error exporting to Anki: {exc}")
            return
        self.status_label.config(text=f"Saved Anki import file to {path} — open it from Anki's File > Import.")

    def _detail_speak(self):
        if self.selected_word:
            self.speak(self.selected_word)

    def _detail_find_all(self):
        if self.selected_word:
            self.open_concordance(self.selected_word)

    def _detail_word_info(self):
        if not self.selected_word:
            return
        WordInfoWindow(
            self, self.selected_word, self.palette, lang=self.detected_lang or "en",
            wordnet_lang=WORDNET_LANG_CODES.get(self.detected_lang), sentences=self.book_sentences,
        )

    def _detail_focus_view(self):
        # word_lookup values are (definition, count, example, pos,
        # level) — see render_results()'s insert_word_row() closure,
        # the same source _show_selection() itself reads from, so this
        # shows exactly what a reader would already see one word at a
        # time, just for the whole selection at once.
        words = [w for w in self.selected_words if w in self.word_lookup]
        if not words:
            return
        entries = []
        for word in words:
            definition, _count, example, pos, level = self.word_lookup[word]
            entries.append((word, pos, level, definition, example))
        FocusViewWindow(
            self, entries, self.palette, on_mark=self._detail_mark,
            on_speak=self.speak, on_study=self._focus_view_study,
        )

    def _focus_view_study(self, mode):
        # Reads self.selected_words fresh, same as _detail_mark() does
        # for marking — Focus View itself never keeps its own copy of
        # "which words", it just hands the mode back once closed (see
        # FocusViewWindow._study()) and this re-reads the same selection
        # the main tree still has, exactly the way on_mark already works.
        words = [w for w in self.selected_words if w in self.word_lookup and w not in self.known_words]
        if not words:
            messagebox.showinfo(
                "Study", "Nothing left to study here — every one of these words is already marked known.",
            )
            return
        entries = [(word, self.word_lookup[word][0]) for word in words]
        StudyWindow(
            self, entries, self.word_examples, on_finish=self._record_study_results,
            lang=self.detected_lang, palette=self.palette, schedule=self.progress_schedule,
            mode=mode, already_known_words=self.known_words, learned_at=self.progress_learned_at,
            pos_map=self.word_pos_map, cefr_map=self.word_cefr_map,
        )

    def _detail_mark(self, status):
        words = self.selected_words
        if not words:
            return
        for word in words:
            if status == "known":
                self.known_words.add(word)
                self.learning_words.discard(word)
            else:
                self.learning_words.add(word)
                self.known_words.discard(word)
            # Marking a word here (not through a Study-mode flashcard)
            # should still count as a real review for scheduling
            # purposes — otherwise a word marked known straight from the
            # table would stay "due" and get shown again immediately in
            # the very next Study session.
            review_word(self.progress_schedule, word, status == "known")
            if status == "known":
                mark_known(self.progress_learned_at, word)
        self._save_progress()
        self._refresh_vocab_stat()
        self._refresh_reading_readiness()
        self._refresh_progress_stat()
        self.render_results()

    def _copy_selection(self, _event=None):
        words = [w for w in self.tree.selection() if w in self.word_lookup]
        if not words:
            return "break"
        lines = []
        for word in words:
            definition, count, example, pos, level = self.word_lookup[word]
            line = f"{word}\t{count or ''}\t{pos or ''}\t{level or ''}\t{definition}"
            if example:
                line += f"\t{example}"
            lines.append(line)
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        plural = "s" if len(words) != 1 else ""
        self.status_label.config(text=f"Copied {len(words)} word{plural} to the clipboard.")
        return "break"

    def _copy_words_only(self):
        words = [w for w in self.tree.selection() if w in self.word_lookup]
        if not words:
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(words))
        plural = "s" if len(words) != 1 else ""
        self.status_label.config(text=f"Copied {len(words)} word{plural} to the clipboard.")

    def _copy_definitions_only(self):
        words = [w for w in self.tree.selection() if w in self.word_lookup]
        if not words:
            return
        definitions = [self.word_lookup[w][0] for w in words]
        self.clipboard_clear()
        self.clipboard_append("\n".join(definitions))
        plural = "s" if len(words) != 1 else ""
        self.status_label.config(text=f"Copied {len(words)} definition{plural} to the clipboard.")

    # -------------------------------------------------------------- chart

    def _draw_chart(self):
        c = self.chart_canvas
        c.delete("all")
        if not self.word_freqs:
            return
        buckets = [("1-10", 0), ("11-50", 0), ("51-100", 0), ("100+", 0)]
        for count in self.word_freqs.values():
            if count <= 10:
                buckets[0] = (buckets[0][0], buckets[0][1] + 1)
            elif count <= 50:
                buckets[1] = (buckets[1][0], buckets[1][1] + 1)
            elif count <= 100:
                buckets[2] = (buckets[2][0], buckets[2][1] + 1)
            else:
                buckets[3] = (buckets[3][0], buckets[3][1] + 1)
        max_val = max(v for _, v in buckets) or 1
        p = self.palette
        width = max(int(c.winfo_width()) or 220, 220)
        row_h = 34
        for i, (label, value) in enumerate(buckets):
            y = i * row_h + 8
            c.create_text(4, y + 10, anchor="w", text=label, fill=p["FG"], font=th.FONT_SMALL_BOLD)
            bar_max = width - 90
            bar_w = int(bar_max * value / max_val)
            c.create_rectangle(60, y, 60 + bar_w, y + 18, fill=p["ACCENT"], width=0)
            c.create_text(66 + bar_w, y + 9, anchor="w", text=str(value), fill=p["DIM"], font=th.FONT_SMALL_BOLD)
        c.config(height=len(buckets) * row_h + 12)

    # ----------------------------------------------------------- study mode

    def open_study_mode(self, mode="flashcard"):
        if not self.glossary_entries:
            return
        query = self.search_var.get().strip().lower()
        entries = [
            e for e in self.glossary_entries
            if (not query or e[0].startswith(query)) and e[0] not in self.known_words
        ]
        if entries:
            StudyWindow(
                self, entries, self.word_examples, on_finish=self._record_study_results,
                lang=self.detected_lang, palette=self.palette, schedule=self.progress_schedule,
                mode=mode, already_known_words=self.known_words, learned_at=self.progress_learned_at,
                pos_map=self.word_pos_map, cefr_map=self.word_cefr_map,
            )
        else:
            messagebox.showinfo("Study", "Nothing left to study here — every matching word is already marked known.")

    # --------------------------------------------------------- find in book

    def open_concordance(self, initial_query=""):
        if not self.book_sentences:
            return
        ConcordanceWindow(self, self.book_sentences, self.palette, initial_query=initial_query or "")

    # -------------------------------------------------------------- saving

    def save_as(self):
        # The footer's SAVE button is shared across both sections — see
        # _refresh_export_buttons() — so it exports whichever section's
        # data is actually on screen right now, rather than always
        # meaning "the vocabulary glossary."
        if self.active_section == "grammar":
            self._save_grammar_as()
        elif self.active_section == "phrases":
            self._save_phrases_as()
        else:
            self._save_vocab_as()

    def _save_vocab_as(self):
        if not self.glossary_entries:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt"), ("Word document", "*.docx"), ("PDF file", "*.pdf"),
                ("CSV file", "*.csv"), ("JSON file", "*.json"), ("Markdown file", "*.md"),
            ],
            initialfile="vocabulary_glossary.txt",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        args = (path, self.filepath, self.glossary_entries, self.undefined_words, self.word_freqs,
                self.word_examples, self.sort_by_freq)
        try:
            if ext == ".docx":
                exp.export_docx(*args, pos_map=self.word_pos_map, cefr_map=self.word_cefr_map)
            elif ext == ".pdf":
                exp.export_pdf(*args, pos_map=self.word_pos_map, cefr_map=self.word_cefr_map)
            elif ext == ".csv":
                exp.export_csv(
                    *args, known_words=self.known_words, learning_words=self.learning_words,
                    pos_map=self.word_pos_map, cefr_map=self.word_cefr_map,
                )
            elif ext == ".json":
                exp.export_json(
                    *args, lang=self.detected_lang, known_words=self.known_words,
                    learning_words=self.learning_words, pos_map=self.word_pos_map, cefr_map=self.word_cefr_map,
                )
            elif ext == ".md":
                exp.export_markdown(*args, pos_map=self.word_pos_map, cefr_map=self.word_cefr_map)
            else:
                exp.export_txt(*args, pos_map=self.word_pos_map, cefr_map=self.word_cefr_map)
        except Exception as exc:
            self.status_label.config(text=f"Error saving: {exc}")
            return
        self.status_label.config(text=f"Saved to {path}")

    def _save_grammar_as(self):
        if not self.grammar_results:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt"), ("Word document", "*.docx"), ("PDF file", "*.pdf"),
                ("CSV file", "*.csv"), ("JSON file", "*.json"), ("Markdown file", "*.md"),
            ],
            initialfile="grammar_analysis.txt",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        args = (path, self.filepath, self.grammar_results, self.grammar_known, self.grammar_learning)
        try:
            if ext == ".docx":
                exp.export_grammar_docx(*args)
            elif ext == ".pdf":
                exp.export_grammar_pdf(*args)
            elif ext == ".csv":
                exp.export_grammar_csv(*args)
            elif ext == ".json":
                exp.export_grammar_json(
                    path, self.filepath, self.grammar_results, self.detected_lang,
                    self.grammar_known, self.grammar_learning,
                )
            elif ext == ".md":
                exp.export_grammar_markdown(*args)
            else:
                exp.export_grammar_txt(*args)
        except Exception as exc:
            self.status_label.config(text=f"Error saving: {exc}")
            return
        self.status_label.config(text=f"Saved to {path}")

    def _save_phrases_as(self):
        if not self.phrases:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt"), ("Word document", "*.docx"), ("PDF file", "*.pdf"),
                ("CSV file", "*.csv"), ("JSON file", "*.json"), ("Markdown file", "*.md"),
            ],
            initialfile="phrase_analysis.txt",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".docx":
                exp.export_phrases_docx(path, self.filepath, self.phrases)
            elif ext == ".pdf":
                exp.export_phrases_pdf(path, self.filepath, self.phrases)
            elif ext == ".csv":
                exp.export_phrases_csv(path, self.filepath, self.phrases)
            elif ext == ".json":
                exp.export_phrases_json(path, self.filepath, self.phrases, self.detected_lang)
            elif ext == ".md":
                exp.export_phrases_markdown(path, self.filepath, self.phrases)
            else:
                exp.export_phrases_txt(path, self.filepath, self.phrases)
        except Exception as exc:
            self.status_label.config(text=f"Error saving: {exc}")
            return
        self.status_label.config(text=f"Saved to {path}")

    def export_anki(self):
        # Same section-aware reuse as save_as() above.
        if self.active_section == "grammar":
            self.export_grammar_anki()
        elif self.active_section == "phrases":
            self._export_phrases_anki()
        else:
            self._export_vocab_anki()

    def _export_vocab_anki(self):
        if not self.glossary_entries:
            return
        deck_name = "Vocabulary"
        if self.filepath:
            deck_name = os.path.splitext(os.path.basename(self.filepath))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Anki-importable text file", "*.txt")],
            initialfile=f"{deck_name}_anki.txt",
        )
        if not path:
            return
        try:
            exp.export_anki(
                path, self.glossary_entries, self.word_freqs, self.word_examples, self.sort_by_freq,
                self.known_words, self.learning_words, deck_name, pos_map=self.word_pos_map,
                cefr_map=self.word_cefr_map,
            )
        except Exception as exc:
            self.status_label.config(text=f"Error exporting to Anki: {exc}")
            return
        self.status_label.config(text=f"Saved Anki import file to {path} — open it from Anki's File > Import.")
