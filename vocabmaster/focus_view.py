"""Focus View: a maximized review window for a multi-word selection in
the Vocabulary tree. The main window's own detail panel only has room
for a one-line hint ("4 words selected...") once more than one row is
selected — this gives the same selection real screen space: every
word's part of speech, CEFR level, definition, and example, large and
readable, so a reader can actually review a batch before deciding
known/still-learning instead of marking blind.

Marking from here does exactly what the main window's own I KNOW IT/
LEARNING buttons already do for a multi-selection — app.py's
_detail_mark() is reused unchanged as the `on_mark` callback, so this
is a bigger, easier-to-read view onto the same action, not a new one.
SPEAK (per word) and FLASHCARDS/WRITING QUIZ (for the whole reviewed
set) are the same pattern: `on_speak`/`on_study` hand straight back to
app.py's own self.speak()/StudyWindow-opening logic, so this window
never re-implements TTS or spaced-repetition scheduling itself — it's
still just a bigger window onto actions that already exist.

The whole body is ONE tk.Text widget (state="disabled", tags for each
piece's styling) rather than a Canvas full of Label widgets — a plain
Label can't be text-selected with the mouse at all, confirmed directly
to matter (a reader wanting to copy a definition or example sentence
out of this window had no way to), while a disabled Text widget already
supports real click-and-drag selection and Ctrl+C copy, same "readonly
but selectable" pattern already used for every other detail panel in
this app (word_info.py, phrase_analyzer.py, grammar_analyzer.py). This
also drops the Canvas-resize/mousewheel-forwarding plumbing the old
per-card-Frame layout needed: a Text widget reflows its own wrapping on
resize and already scrolls with the mouse wheel natively. The one place
that still needs a real embedded widget rather than plain text is each
word's SPEAK icon (window_create — Text widgets support embedding an
actual interactive widget inline, unlike a Label-only layout, so this
doesn't lose the "no way to trigger anything" gap a static Canvas of
Labels would still have).
"""
import tkinter as tk
from tkinter import ttk

import theme as th


class FocusViewWindow(tk.Toplevel):
    def __init__(self, master, entries, palette, on_mark, on_speak=None, on_study=None):
        """`entries` is [(word, pos, level, definition, example), ...]
        for the words to review, already in whatever order the caller
        sorted them (app.py passes them in the same order the tree
        shows them, not re-sorted here). `on_mark` is called with
        "known"/"learning" and takes no other arguments — the caller
        already knows which words are selected (self.selected_words),
        same division of responsibility the main window's own
        multi-select marking already uses. `on_speak`, if given, is
        called with a single word to pronounce it (app.py's own
        self.speak — same threaded TTS the main Detail panel's SPEAK
        button already uses); omitting it simply hides every word's
        speaker icon rather than showing one that does nothing.
        `on_study`, if given, is called with "flashcard" or "writing" to
        launch a Study session scoped to this same word selection —
        closes this window first (see _study()), the same "hand off,
        don't stack two full-screen review windows" choice as marking
        known/learning already makes.
        """
        super().__init__(master)
        self.p = palette
        self.title(f"Focus View — {len(entries)} words")
        self.configure(bg=self.p["BG"])
        # Maximized, not borderless-fullscreen (tk's "-fullscreen"
        # attribute) — keeps the window's own title bar/close button and
        # taskbar presence, so it behaves like any other window the
        # reader can still Alt-Tab away from or drag, just starting at
        # full size rather than a fixed small popup geometry.
        self.state("zoomed")

        header = tk.Frame(self, bg=self.p["BG"])
        header.pack(fill="x", padx=30, pady=(20, 10))
        tk.Label(
            header, text=f"REVIEWING {len(entries)} SELECTED WORDS", bg=self.p["BG"], fg=self.p["FG"],
            font=th.FONT_TITLE,
        ).pack(side="left")
        close_btn = ttk.Button(header, text="CLOSE (Esc)", command=lambda: self._close())
        close_btn.pack(side="right")

        body_frame = tk.Frame(
            self, bg=self.p["PANEL"], highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"],
        )
        body_frame.pack(fill="both", expand=True, padx=30, pady=(0, 10))
        scrollbar = tk.Scrollbar(
            body_frame, bg=self.p["PANEL"], troughcolor=self.p["BG"], activebackground=self.p["SECOND_HOVER"],
            highlightthickness=0,
        )
        scrollbar.pack(side="right", fill="y")
        text = tk.Text(
            body_frame, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word", relief="flat",
            yscrollcommand=scrollbar.set, highlightthickness=0, padx=24, pady=16,
            cursor="xterm", insertbackground=self.p["FG"],
            selectbackground=self.p["ACCENT"], selectforeground=self.p["ACCENT_TEXT"],
        )
        text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text.yview)

        text.tag_configure("word", font=th.FONT_CARD_WORD, foreground=self.p["ACCENT"], spacing1=18)
        text.tag_configure("meta", font=th.FONT_SMALL_ITALIC, foreground=self.p["DIM"])
        text.tag_configure("def", font=th.FONT, foreground=self.p["FG"], spacing1=6)
        text.tag_configure("example", font=th.FONT_SMALL_ITALIC, foreground=self.p["RUBRIC"], spacing1=6)
        # Confirmed directly this matters, not just in theory: Tk's
        # built-in "sel" tag is created before any custom tag, which
        # makes it the LOWEST-priority tag by Tk's own default stacking
        # order — a later tag's own `foreground` (here, "word"'s own
        # accent color) wins over sel's selectforeground for whichever
        # attribute both set, so selecting the word rendered ACCENT-on-
        # ACCENT-selection came out invisible (selectbackground painted
        # behind ACCENT-colored text that "sel" could no longer
        # recolor). Raising "sel" to the top priority is the standard
        # fix: it guarantees the selection highlight always wins,
        # however this widget's other tags are colored, rather than
        # only working by coincidence for tags that don't happen to
        # collide with a specific selectbackground color.
        text.tag_raise("sel")
        # A real embedded hairline (window_create) doesn't stretch to the
        # Text widget's own width without the same resize-tracking
        # plumbing the old Canvas layout needed just for wraplength —
        # this dim rule character (repeated, not a single thin line)
        # gets the same "entries are visually separated" effect without
        # it, and reflows naturally with the rest of the text on resize.
        text.tag_configure("rule", foreground=self.p["PANEL_BORDER"], spacing1=14, spacing3=4)

        for i, (word, pos, level, definition, example) in enumerate(entries):
            if i > 0:
                text.insert("end", "─" * 60 + "\n", "rule")
            text.insert("end", word, "word")
            meta = " · ".join(m for m in (pos, level) if m)
            if meta:
                text.insert("end", f"   {meta}", "meta")
            if on_speak:
                text.insert("end", " ")
                # A small flat tk.Button (not ttk's "Icon.TButton" —
                # that style hardcodes a "BG" background, which would
                # show as a mismatched box against this Text widget's
                # own "PANEL" background) embedded inline right after
                # the word — window_create, not a second widget layout,
                # so it sits in the text flow and still scrolls/wraps
                # with everything else.
                speak_btn = tk.Button(
                    text, text="\U0001F50A", command=lambda w=word: on_speak(w),
                    bg=self.p["PANEL"], activebackground=self.p["SECOND_HOVER"], relief="flat", bd=0,
                    font=th.FONT_SMALL_BOLD, cursor="hand2", padx=4, pady=0,
                )
                text.window_create("end", window=speak_btn)
            text.insert("end", "\n")
            text.insert("end", definition, "def")
            if example:
                text.insert("end", "\n")
                text.insert("end", f"“{example}”", "example")
            text.insert("end", "\n")

        text.config(state="disabled")

        footer = tk.Frame(self, bg=self.p["BG"])
        footer.pack(fill="x", padx=30, pady=(0, 24))
        ttk.Button(
            footer, text="\U0001F4D6 MARK ALL STILL LEARNING", command=lambda: self._mark(on_mark, "learning"),
        ).pack(side="left")
        ttk.Button(
            footer, text="✅ MARK ALL KNOWN", command=lambda: self._mark(on_mark, "known"),
            style="Accent.TButton",
        ).pack(side="left", padx=(8, 0))
        if on_study:
            ttk.Button(
                footer, text="✏ WRITING QUIZ", command=lambda: self._study(on_study, "writing"),
            ).pack(side="right")
            ttk.Button(
                footer, text="\U0001F4D6 FLASHCARDS", command=lambda: self._study(on_study, "flashcard"),
                style="Accent.TButton",
            ).pack(side="right", padx=(0, 8))

        self.bind("<Escape>", lambda _e: self._close())
        th.apply_dark_titlebar(self, self.p is th.DARK)

    def _mark(self, on_mark, status):
        on_mark(status)
        self._close()

    def _study(self, on_study, mode):
        self._close()
        on_study(mode)

    def _close(self):
        self.destroy()
