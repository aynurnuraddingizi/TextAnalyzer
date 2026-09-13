"""Flashcard Study mode (moved from the original vocabulary_app.py) plus
a small docked "ready to study" summary card used in the main window's
right-hand column.

Full keyboard shortcuts (Space/1/2/S). The word/definition/example are
real selectable text (a readonly Entry + Text widget, not Labels) so a
reader can drag-select and copy any of it — this replaced an earlier
click-anywhere-on-the-word-to-speak shortcut, which would have fought
with using that same click to position a cursor or start a selection;
the SPEAK button is the one way to hear it now.

No Skip: removed by request — every card gets a real known/still-
learning verdict now, rather than an option to leave one unrecorded.

No translate-to-English feature: tried and removed, see app.py's module
docstring for why.
"""
import random
import threading
import tkinter as tk
from tkinter import ttk

from text_analyzer import due_words, mark_known, mask_word_in_sentence, review_word, speak_word

import theme as th
from export import combined_tag


class StudyWindow(tk.Toplevel):
    """Shows a word, reveals its definition on demand, and sorts each
    card into a "known" or "still learning" outcome. Ending the session
    — finishing the deck or clicking END SESSION early — shows the full
    word list for known/learning, not just a
    tally.
    """

    def __init__(self, master, entries, word_examples=None, on_finish=None, lang="en", palette=None,
                 schedule=None, mode="flashcard", already_known_words=None, learned_at=None, pos_map=None,
                 cefr_map=None):
        super().__init__(master)
        self.p = palette or th.LIGHT
        # "flashcard": see the word, self-report know-it/still-learning.
        # "writing": see the definition and a blanked-out example, TYPE
        # the word, and the verdict is decided automatically by whether
        # the typed answer matches — active recall rather than passive
        # recognition, and it removes the temptation to mark a word
        # "known" just for having recognized it a moment ago.
        self.mode = mode if mode in ("flashcard", "writing") else "flashcard"
        self.title("Study — Text Analyzer")
        self.geometry("640x520")
        # Measured directly: this layout's own natural width/height is
        # 498x375 — minsize sits comfortably above that (not just above
        # the old, wider layout's requirement) so shrinking the window as
        # far as it'll go still can't clip a button off the edge again.
        self.minsize(560, 420)
        self.configure(bg=self.p["BG"])
        self.word_examples = word_examples or {}
        # word -> "noun"/"verb"/etc., or simply absent for a word with
        # no part-of-speech data at all (every offline-dictionary-only
        # language — see build_pos_map()'s docstring in text_analyzer.py).
        self.pos_map = pos_map or {}
        # word -> "A1".."C2", English-only — see build_cefr_map()'s
        # docstring in text_analyzer.py.
        self.cefr_map = cefr_map or {}
        self.lang = lang or "en"
        # Called once, when the session ends (naturally or via END
        # SESSION), with the plain word lists so the main app can persist
        # them to long-term progress — not per-card, since a card's
        # known/learning status can still change later in the same
        # session if the reader reconsiders.
        self.on_finish = on_finish
        # The same dict instance the app loaded from disk — mutated
        # in-place by next_card() below via review_word(), so by the time
        # on_finish fires it already reflects this whole session's SM-2
        # updates with no separate hand-back needed. None (a book opened
        # before spaced repetition existed some other way) just turns
        # every review into a no-op rather than an error.
        self.schedule = schedule
        # Same dict-mutated-in-place pattern as self.schedule — records
        # the first time each word is ever marked known, so the Progress
        # card's "this week"/"this month" stats (words_learned_since())
        # have something to measure. None turns it into a no-op, same as
        # self.schedule does.
        self.learned_at = learned_at
        # The SAME set object app.py's MainWindow keeps as self.known_words
        # — not a copy — so a word marked known elsewhere (the glossary
        # table's Detail panel) WHILE this session is already open and
        # running is still seen: every mutation there is `.add()`/`|=`
        # in place, never a reassignment, so this reference stays live.
        # Without this, a session opened before marking a word known took
        # a one-time snapshot of the word list and kept offering that
        # word as a card regardless of what happened in the main window
        # afterward. The initial `entries` passed in are already filtered
        # by app.py at the moment this window opens; this only matters
        # for words that become known mid-session, from here on.
        self.already_known_words = already_known_words if already_known_words is not None else set()
        # Cards due for review (never studied, or past their SM-2 due
        # date) go first, each group separately shuffled — the reader
        # sees what's actually worth reviewing today before the rest,
        # instead of a flat random shuffle treating every word the same
        # regardless of how recently it was last seen. Nothing is
        # excluded: a not-yet-due word still shows up, just later.
        entries = list(entries)
        if self.schedule is not None:
            due = due_words(self.schedule, [w for w, _d in entries])
            due_entries = [e for e in entries if e[0] in due]
            later_entries = [e for e in entries if e[0] not in due]
        else:
            due_entries, later_entries = entries, []
        random.shuffle(due_entries)
        random.shuffle(later_entries)
        self.cards = due_entries + later_entries
        self.index = 0
        self.known_words = []      # [(word, definition), ...]
        self.learning_words = []   # [(word, definition), ...]
        self.revealed = False
        self.finished = False
        self._last_known = False  # writing mode: verdict from the most recent check_answer()

        self.counter_label = tk.Label(self, text="", bg=self.p["BG"], fg=self.p["DIM"], font=th.FONT)
        self.counter_label.pack(pady=(20, 0))

        self.card_frame = tk.Frame(
            self, bg=self.p["PANEL"], highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"],
        )
        self.card_frame.pack(fill="both", expand=True, padx=30, pady=20)

        # A readonly Entry, not a Label: state="readonly" blocks typing
        # but still allows mouse drag-to-select and Ctrl+C/right-click
        # Copy, which a Label can't do at all — confirmed directly this
        # was needed, the reader wants to copy the word itself (e.g. to
        # paste into a translator elsewhere), same fix as the Detail
        # panel's word field in the main window. This drops the old
        # click-anywhere-on-the-word-to-speak shortcut (a single click
        # now positions the text cursor / starts a selection instead,
        # which would otherwise fight with that) — the SPEAK button
        # below remains the actual way to hear it.
        self.word_var = tk.StringVar()
        self.word_entry = tk.Entry(
            self.card_frame, textvariable=self.word_var, state="readonly",
            readonlybackground=self.p["PANEL"], fg=self.p["ACCENT"], font=th.FONT_CARD_WORD,
            relief="flat", highlightthickness=0, bd=0, justify="center", cursor="xterm",
            insertbackground=self.p["ACCENT"], selectbackground=self.p["ACCENT"],
            selectforeground=self.p["ACCENT_TEXT"],
        )

        # Definition and example share one Text widget (same reasoning,
        # and the same pattern, as the main window's Detail panel): real
        # cursor drag-to-select and copy, which neither a Label nor a
        # disabled Text widget's separate-labels version could do. The
        # example is a real sentence from the book itself using the
        # word, shown alongside the definition on reveal — genuine
        # usage, not an isolated word + translation, is what actually
        # helps retention.
        self.card_text = tk.Text(
            self.card_frame, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word",
            relief="flat", highlightthickness=0, padx=0, pady=0, borderwidth=0,
            cursor="xterm", insertbackground=self.p["FG"], height=5,
        )
        self.card_text.tag_configure("def", foreground=self.p["FG"], font=th.FONT, justify="center")
        self.card_text.tag_configure("pos", foreground=self.p["DIM"], font=th.FONT_SMALL_ITALIC, justify="center")
        self.card_text.tag_configure("example", foreground=self.p["RUBRIC"], font=th.FONT_SMALL_ITALIC, justify="center")
        # Writing-mode feedback after checking an answer — reusing
        # ACCENT (already this app's "positive/primary" color, e.g. the
        # I KNOW IT button) and RUBRIC (already used for attention-
        # drawing text, e.g. examples) rather than adding new palette
        # entries just for this.
        self.card_text.tag_configure("correct", foreground=self.p["ACCENT"], font=th.FONT_BOLD, justify="center")
        self.card_text.tag_configure("incorrect", foreground=self.p["RUBRIC"], font=th.FONT_BOLD, justify="center")
        self.card_text.config(state="disabled")

        # Writing mode only: a real (editable, not readonly) Entry for
        # typing the answer, plus a CHECK button that becomes NEXT once
        # an answer's been submitted — same "repurpose the button rather
        # than build a second one" pattern show_summary() already uses
        # on END SESSION -> CLOSE below.
        self.answer_var = tk.StringVar()
        self.answer_entry = tk.Entry(
            self.card_frame, textvariable=self.answer_var, bg=self.p["BG"], fg=self.p["FG"],
            insertbackground=self.p["FG"], font=th.FONT_ROW_WORD, justify="center", relief="flat",
            highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"], highlightcolor=self.p["ACCENT"],
        )
        self.answer_entry.bind("<Return>", lambda _e: self._on_writing_enter())

        # Summary view — built once, shown in place of the card labels
        # once the session ends.
        self.summary_frame = tk.Frame(self.card_frame, bg=self.p["PANEL"])
        summary_scroll = tk.Scrollbar(
            self.summary_frame, bg=self.p["PANEL"], troughcolor=self.p["BG"],
            activebackground=self.p["SECOND_HOVER"], highlightthickness=0,
        )
        summary_scroll.pack(side="right", fill="y")
        self.summary_text = tk.Text(
            self.summary_frame, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word",
            relief="flat", yscrollcommand=summary_scroll.set, highlightthickness=0, padx=10, pady=10,
        )
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scroll.config(command=self.summary_text.yview)
        self.summary_text.tag_configure("heading", font=th.FONT_ROW_WORD, foreground=self.p["ACCENT"])
        self.summary_text.tag_configure("known", foreground=self.p["FG"], font=th.FONT)
        self.summary_text.tag_configure("learning", foreground=self.p["RUBRIC"], font=th.FONT)
        self.summary_text.config(state="disabled")

        # Three rows, grouped by purpose rather than one wide row per
        # action — confirmed directly that the old 4-buttons-in-one-row
        # nav bar (STILL LEARNING/SKIP/END SESSION/I KNOW IT) clipped
        # past the window's edge if the reader resized the (resizable)
        # window narrower than that row's natural width, since Tkinter's
        # pack geometry manager doesn't reflow/wrap content that no
        # longer fits — it just extends past the visible edge instead.
        # Reveal alone on top (it gates everything else), the actual
        # know/learning verdict as an equal-width pair in the middle
        # (the two actions actually used every single card), and the
        # least-used actions (speak, end) in a smaller row at the
        # bottom — this halves the widest row's button count, and
        # `minsize` below is set from this layout's own measured width
        # plus margin so the clipping bug can't recur even if resized
        # down to the smallest the window allows. Skip (advance without
        # recording either way) was removed by request — every card now
        # gets a real known/still-learning verdict.
        #
        # Writing mode swaps this top slot for CHECK ANSWER (the
        # answer_entry built above sits in card_frame, right below the
        # prompt) and never shows verdict_row at all — the verdict is
        # decided automatically from whether the typed answer matches,
        # not chosen by the reader.
        self.reveal_btn = ttk.Button(
            self, text="SHOW ANSWER (space)", command=self.reveal, style="Accent.TButton"
        )
        self.check_btn = ttk.Button(
            self, text="CHECK ANSWER (enter)", command=self.check_answer, style="Accent.TButton"
        )
        if self.mode == "writing":
            self.check_btn.pack(fill="x", padx=30, pady=(0, 8))
        else:
            self.reveal_btn.pack(fill="x", padx=30, pady=(0, 8))

        self.verdict_row = tk.Frame(self, bg=self.p["BG"])
        self.learning_btn = ttk.Button(
            self.verdict_row, text="STILL LEARNING (2)", command=lambda: self.next_card(False)
        )
        self.learning_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.known_btn = ttk.Button(
            self.verdict_row, text="I KNOW IT (1)", command=lambda: self.next_card(True), style="Accent.TButton"
        )
        self.known_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))
        if self.mode == "flashcard":
            self.verdict_row.pack(fill="x", padx=30, pady=(0, 8))

        utility_row = tk.Frame(self, bg=self.p["BG"])
        utility_row.pack(fill="x", padx=30, pady=(0, 20))
        self.speak_btn = ttk.Button(utility_row, text="SPEAK (s)", command=self._speak_current)
        self.speak_btn.pack(side="left")
        self.end_btn = ttk.Button(utility_row, text="END SESSION", command=self.end_session)
        self.end_btn.pack(side="right")

        if self.mode == "flashcard":
            self.bind("<space>", lambda _e: self.reveal())
            self.bind("<Return>", lambda _e: self.next_card(True))
            self.bind("<Key-1>", lambda _e: self.next_card(True))
            self.bind("<Key-2>", lambda _e: self.next_card(False))
        else:
            # answer_entry has its own <Return> binding above (fires
            # first while it has focus); this window-level one is the
            # fallback for when it doesn't (e.g. focus already moved
            # after a mouse-click CHECK). Both routes call the same
            # context-aware handler, so there's exactly one place that
            # decides "submit" vs "next".
            self.bind("<Return>", lambda _e: self._on_writing_enter())
        self.bind("<Key-s>", lambda _e: self._speak_current())
        self.bind("<Key-S>", lambda _e: self._speak_current())
        th.apply_dark_titlebar(self, self.p is th.DARK)
        self.show_card()
        self.focus_set()

    def _speak_current(self):
        if self.finished or self.index >= len(self.cards):
            return
        # Writing mode only: speaking the word before it's been checked
        # would just hand the reader the answer, defeating the point of
        # a recall quiz. Flashcard mode already showed the word from the
        # start (that's the point there), so speaking it any time —
        # before or after SHOW ANSWER — was always fine and stays so.
        if self.mode == "writing" and not self.revealed:
            return
        word, _ = self.cards[self.index]
        threading.Thread(target=self._run_speak, args=(word,), daemon=True).start()

    def _run_speak(self, word):
        try:
            speak_word(word, self.lang)
        except Exception:
            # A failed pronunciation (e.g. no internet) shouldn't
            # interrupt studying — this is a bonus, not worth an error
            # dialog mid-quiz.
            pass

    def show_card(self):
        # Drop (not just skip past) any card whose word was marked known
        # elsewhere since this session opened — removing it outright,
        # rather than leaving it in place and stepping over it, keeps
        # len(self.cards) (what the "X/Y" counter and end-of-session
        # "not reached" count are both based on) honest about what this
        # session actually still has left to show.
        while self.index < len(self.cards) and self.cards[self.index][0] in self.already_known_words:
            del self.cards[self.index]
        if self.index >= len(self.cards):
            self.show_summary()
            return
        self.summary_frame.pack_forget()
        self.card_text.pack(pady=10, padx=20, fill="x")
        word, definition = self.cards[self.index]
        self.card_text.config(state="normal")
        self.card_text.delete("1.0", "end")
        if self.mode == "writing":
            # The word itself stays hidden until check_answer() reveals
            # it — showing the prompt (definition + example, with the
            # answer blanked out of the example) is the whole point here,
            # unlike flashcard mode where the word is the starting point.
            self.word_entry.pack_forget()
            self.word_var.set("")
            tag_label = combined_tag(word, self.pos_map, self.cefr_map)
            if tag_label:
                self.card_text.insert("end", f"({tag_label})  ", "pos")
            self.card_text.insert("end", definition, "def")
            example = self.word_examples.get(word)
            if example:
                masked = mask_word_in_sentence(example, word)
                self.card_text.insert("end", f"\n\n“{masked}”", "example")
            self.answer_var.set("")
            self.answer_entry.pack(pady=(0, 10), padx=20, fill="x")
            self.answer_entry.config(state="normal")
            self.answer_entry.focus_set()
            self.check_btn.config(text="CHECK ANSWER (enter)", command=self.check_answer, state="normal")
            self.speak_btn.config(state="disabled")
        else:
            self.word_entry.pack(pady=(40, 10))
            self.word_var.set(word)
            self.answer_entry.pack_forget()
        self.card_text.config(state="disabled")
        self.revealed = False
        self.reveal_btn.config(state="normal")
        self.counter_label.config(text=f"{self.index + 1}/{len(self.cards)}")

    def reveal(self):
        if self.finished or self.revealed:
            return
        word, definition = self.cards[self.index]
        self.card_text.config(state="normal")
        tag_label = combined_tag(word, self.pos_map, self.cefr_map)
        if tag_label:
            self.card_text.insert("end", f"({tag_label})  ", "pos")
        self.card_text.insert("end", definition, "def")
        example = self.word_examples.get(word)
        if example:
            self.card_text.insert("end", f"\n\n“{example}”", "example")
        self.card_text.config(state="disabled")
        self.revealed = True

    def check_answer(self):
        """Writing mode only: grade the typed answer against the actual
        word (case-insensitive, exact match — no partial credit) and
        reveal the correct word either way. The verdict this produces
        feeds next_card() exactly like a flashcard's self-reported one
        does, so it gets the same known/learning bookkeeping and SM-2
        scheduling for free.
        """
        if self.finished or self.revealed:
            return
        word, _definition = self.cards[self.index]
        typed = self.answer_var.get().strip()
        known = typed.lower() == word.lower()
        self._last_known = known

        self.word_var.set(word)
        self.word_entry.pack(pady=(20, 10), before=self.card_text)

        self.card_text.config(state="normal")
        if known:
            self.card_text.insert("end", "\n\n✅ Correct!", "correct")
        elif typed:
            self.card_text.insert("end", f"\n\n❌ You wrote “{typed}” — not quite.", "incorrect")
        else:
            self.card_text.insert("end", "\n\n❌ No answer given.", "incorrect")
        self.card_text.config(state="disabled")

        self.answer_entry.config(state="disabled")
        self.check_btn.config(text="NEXT → (enter)", command=lambda: self.next_card(known))
        self.speak_btn.config(state="normal")
        self.revealed = True

    def _on_writing_enter(self):
        if self.finished:
            return
        if self.revealed:
            self.next_card(self._last_known)
        else:
            self.check_answer()

    def next_card(self, known):
        if self.finished:
            return
        word, definition = self.cards[self.index]
        (self.known_words if known else self.learning_words).append((word, definition))
        if self.schedule is not None:
            review_word(self.schedule, word, known)
        if known and self.learned_at is not None:
            mark_known(self.learned_at, word)
        self.index += 1
        self.show_card()

    def end_session(self):
        if not self.finished:
            self.show_summary()

    def show_summary(self):
        self.finished = True
        self.word_entry.pack_forget()
        self.card_text.pack_forget()
        self.answer_entry.pack_forget()
        self.summary_frame.pack(fill="both", expand=True, padx=6, pady=6)

        if self.on_finish:
            self.on_finish(
                [w for w, _ in self.known_words],
                [w for w, _ in self.learning_words],
            )

        reviewed = len(self.known_words) + len(self.learning_words)
        not_reached = len(self.cards) - reviewed
        self.counter_label.config(text=f"{reviewed}/{len(self.cards)} reviewed")
        self.reveal_btn.config(state="disabled")
        self.check_btn.config(state="disabled")
        self.speak_btn.config(state="disabled")
        self.learning_btn.config(state="disabled")
        self.known_btn.config(state="disabled")
        self.end_btn.config(text="CLOSE", command=self.destroy)

        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        header = f"SESSION SUMMARY — {reviewed} reviewed"
        if not_reached:
            # Only possible now by clicking END SESSION before finishing
            # the deck — there's no longer a Skip action to cause this.
            header += f", {not_reached} not reached"
        self.summary_text.insert("end", header + "\n\n", "heading")

        self.summary_text.insert("end", f"I KNOW IT  ({len(self.known_words)})\n", "heading")
        if self.known_words:
            for word, _ in sorted(self.known_words):
                self.summary_text.insert("end", f"  {word}\n", "known")
        else:
            self.summary_text.insert("end", "  (none)\n", "known")

        self.summary_text.insert("end", "\n")
        self.summary_text.insert("end", f"STILL LEARNING  ({len(self.learning_words)})\n", "heading")
        if self.learning_words:
            for word, _ in sorted(self.learning_words):
                self.summary_text.insert("end", f"  {word}\n", "learning")
        else:
            self.summary_text.insert("end", "  (none)\n", "learning")
        self.summary_text.config(state="disabled")


def build_study_summary_card(parent, palette, ready_count, on_start, on_export_anki, themed=None):
    """The docked, always-visible "reading readiness" card used in the
    main window's right column: the current book's title/language/word
    count, what fraction of its vocabulary is already known (with a
    progress bar), and the buttons that act on it — not a second
    flashcard implementation, just a live summary plus entry points into
    the real StudyWindow above (one button per mode, see StudyWindow's
    `mode`) and into Anki export.

    `on_start` is called with a single "flashcard"/"writing" argument;
    `on_export_anki` takes none.

    Returns (frame, widgets) — `widgets` is a dict of the pieces the
    caller needs to keep updating as the loaded book/progress changes
    (book_label, meta_label, readiness_label, progress_bar, known_label,
    to_learn_label, count_label, flashcard_btn, writing_btn, anki_btn),
    a dict rather than a long positional tuple since there are enough of
    them now that position would be easy to get wrong.

    `themed`, if given (the main window's ThemedWidgetRegistry), gets
    this card's plain tk widgets registered so a theme toggle actually
    reaches them — without this they were built once with a color
    snapshot and then silently stayed on whichever theme was active at
    launch, one of a few spots that weren't really "shifting" on toggle.
    """
    p = palette
    frame = tk.Frame(parent, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
    header = tk.Label(frame, text="READING READINESS", bg=p["PANEL"], fg=p["FG"], font=th.FONT_BOLD)
    header.pack(anchor="w", padx=12, pady=(10, 0))

    book_label = tk.Label(
        frame, text="No book loaded yet.", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD,
        wraplength=260, justify="left", anchor="w",
    )
    book_label.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
    meta_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT_SMALL_ITALIC, anchor="w")
    meta_label.pack(anchor="w", padx=12, pady=(0, 8))

    readiness_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["ACCENT"], font=th.FONT_SMALL_BOLD, anchor="w")
    readiness_label.pack(anchor="w", padx=12)
    progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=100, value=0)
    progress_bar.pack(fill="x", padx=12, pady=(4, 8))

    known_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["FG"], font=th.FONT, anchor="w")
    known_label.pack(anchor="w", padx=12)
    to_learn_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["FG"], font=th.FONT, anchor="w")
    to_learn_label.pack(anchor="w", padx=12, pady=(0, 10))

    count_label = tk.Label(
        frame, text=f"{ready_count} word(s) ready to study", bg=p["PANEL"], fg=p["DIM"], font=th.FONT,
    )
    count_label.pack(anchor="w", padx=12, pady=(0, 6))
    flashcard_btn = ttk.Button(
        frame, text="FLASHCARDS", command=lambda: on_start("flashcard"), style="Accent.TButton",
    )
    flashcard_btn.pack(padx=12, pady=(0, 6), fill="x")
    writing_btn = ttk.Button(frame, text="✏ WRITING QUIZ", command=lambda: on_start("writing"))
    writing_btn.pack(padx=12, pady=(0, 6), fill="x")
    anki_btn = ttk.Button(frame, text="EXPORT TO ANKI", command=on_export_anki)
    anki_btn.pack(padx=12, pady=(0, 12), fill="x")

    if themed is not None:
        themed.add(frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        themed.add(header, bg="PANEL", fg="FG")
        themed.add(book_label, bg="PANEL", fg="FG")
        themed.add(meta_label, bg="PANEL", fg="DIM")
        themed.add(readiness_label, bg="PANEL", fg="ACCENT")
        themed.add(known_label, bg="PANEL", fg="FG")
        themed.add(to_learn_label, bg="PANEL", fg="FG")
        themed.add(count_label, bg="PANEL", fg="DIM")

    widgets = {
        "book_label": book_label, "meta_label": meta_label, "readiness_label": readiness_label,
        "progress_bar": progress_bar, "known_label": known_label, "to_learn_label": to_learn_label,
        "count_label": count_label, "flashcard_btn": flashcard_btn, "writing_btn": writing_btn,
        "anki_btn": anki_btn,
    }
    return frame, widgets
