"""Grammar Analyzer ("Section 2") UI: browsable category/rule list for
whatever analyze_grammar() (grammar_engine.py) found in a loaded book's
sentences — each rule with a plain-language definition, the underlying
rule, and every matching sentence — plus GrammarQuizWindow (flashcard-
style review) and the docked "grammar readiness" sidebar card.

The actual detection logic (the _RULES catalog, analyze_grammar(),
build_sentence_index(), CATEGORY_ORDER/CATEGORY_ORDER_ES,
ensure_pos_tagger()) lives in grammar_engine.py, not here — that split
exists so the Android app (mobile_app/, which has no Tkinter at all)
can import the detection engine without dragging in this Tkinter UI.
Everything re-exported below (`from grammar_engine import ...`) keeps
app.py's own `from grammar_analyzer import (...)` working unchanged.
"""
import random
import tkinter as tk
from tkinter import ttk

from grammar_engine import (
    CATEGORY_ORDER,
    CATEGORY_ORDER_ES,
    analyze_grammar,
    build_sentence_index,
    ensure_pos_tagger,
)
from text_analyzer import due_words, mark_known, review_word

import theme as th


def build_grammar_section(parent, palette, themed, on_select_rule, on_mark):
    """Builds Section 2's own master-detail layout: a category/rule
    Treeview on the left, a scrollable definition+examples panel on the
    right — a left/right split, unlike the vocabulary section's
    top/bottom split, since a rule's matching sentences need real
    vertical room (potentially dozens of them), not a 4-line blurb.

    `on_select_rule` is called with the selected rule's id (or None,
    when the selection is cleared) — app.py owns the actual data
    (self.grammar_results) and populates the detail widgets itself, the
    same division of responsibility render_results()/_show_selection()
    already use for the vocabulary Treeview. `on_mark` is called with
    "known"/"learning" from the two buttons below the detail text,
    mirroring the vocab detail panel's detail_know_btn/
    detail_learning_btn — app.py owns updating grammar_known/
    grammar_learning and re-rendering afterward, same division again.

    Returns (frame, widgets) — widgets = {"tree", "detail_header_var",
    "detail_text", "know_btn", "learning_btn"} — same (frame,
    widgets)-dict convention as study.build_study_summary_card().
    """
    p = palette
    frame = tk.Frame(parent, bg=p["BG"])

    # Wide enough for the longest real rule name ("Present Perfect
    # Continuous", "Present Participle Phrases", 26 characters) plus its
    # tree-hierarchy indent (a rule row sits one level under its category
    # row, which eats into the "#0" column's usable text width beyond
    # the column's own nominal size) without clipping — confirmed
    # directly: 230px/340px were both still cutting names off
    # mid-word, not just the header label this comment used to be about.
    list_frame = tk.Frame(frame, bg=p["BG"], width=400)
    list_frame.pack(side="left", fill="y", padx=(0, 10))
    list_frame.pack_propagate(False)
    themed.add(list_frame, bg="BG")

    # Shorter text plus a wraplength safety net (wrapping to a second
    # line reads fine; silent clipping doesn't) rather than relying on
    # width tuning alone to always be enough at every DPI setting.
    header = tk.Label(
        list_frame, text="GRAMMAR STRUCTURES", bg=p["BG"], fg=p["FG"], font=th.FONT_BOLD,
        wraplength=380, justify="left", anchor="w",
    )
    header.pack(anchor="w", fill="x", pady=(0, 6))
    themed.add(header, bg="BG", fg="FG")

    tree_frame = tk.Frame(list_frame, bg=p["BG"])
    tree_frame.pack(fill="both", expand=True)
    themed.add(tree_frame, bg="BG")
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
    scrollbar.pack(side="right", fill="y")
    tree = ttk.Treeview(
        tree_frame, columns=("count",), show="tree headings",
        yscrollcommand=scrollbar.set, selectmode="browse",
    )
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=tree.yview)
    tree.heading("#0", text="Structure")
    # "Count", not "Sentences" — matches the vocabulary tree's own count
    # column header exactly, and comfortably fits this narrower column.
    tree.heading("count", text="Count")
    tree.column("#0", width=320, anchor="w", stretch=False)
    tree.column("count", width=60, anchor="center", stretch=False)
    tree.tag_configure("category", foreground=p["ACCENT"], font=th.FONT_SMALL_BOLD)
    # Same status-color roles the vocabulary tree's "new"/"learning"/
    # "known" tags already use, applied here too now that grammar rules
    # get the same know/learning marking.
    tree.tag_configure("new", foreground=p["ACCENT"], font=th.FONT_ROW_WORD)
    tree.tag_configure("learning", foreground=p["SECOND_FG"], font=th.FONT_ROW_WORD)
    tree.tag_configure("known", foreground=p["DIM"], font=th.FONT)

    detail_frame = tk.Frame(frame, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
    detail_frame.pack(side="left", fill="both", expand=True)
    themed.add(detail_frame, bg="PANEL", highlightbackground="PANEL_BORDER")

    detail_header_var = tk.StringVar(value="Select a grammar structure to see details")
    detail_header = tk.Entry(
        detail_frame, textvariable=detail_header_var, state="readonly",
        readonlybackground=p["PANEL"], fg=p["ACCENT"], font=th.FONT_ROW_WORD,
        relief="flat", highlightthickness=0, bd=0, insertbackground=p["ACCENT"],
        selectbackground=p["ACCENT"], selectforeground=p["ACCENT_TEXT"], cursor="xterm",
    )
    detail_header.pack(fill="x", padx=14, pady=(10, 6))
    themed.add(
        detail_header, bg="PANEL", fg="ACCENT", readonlybackground="PANEL",
        selectbackground="ACCENT", selectforeground="ACCENT_TEXT",
    )

    text_row = tk.Frame(detail_frame, bg=p["PANEL"])
    text_row.pack(fill="both", expand=True, padx=14, pady=(0, 10))
    themed.add(text_row, bg="PANEL")
    detail_scrollbar = tk.Scrollbar(
        text_row, bg=p["PANEL"], troughcolor=p["BG"], activebackground=p["SECOND_HOVER"], highlightthickness=0,
    )
    detail_scrollbar.pack(side="right", fill="y")
    detail_text = tk.Text(
        text_row, bg=p["PANEL"], fg=p["FG"], font=th.FONT, wrap="word", relief="flat",
        yscrollcommand=detail_scrollbar.set, highlightthickness=0, padx=0, pady=0,
        cursor="xterm", insertbackground=p["FG"],
    )
    detail_text.pack(side="left", fill="both", expand=True)
    detail_scrollbar.config(command=detail_text.yview)
    detail_text.tag_configure("def", foreground=p["FG"], font=th.FONT)
    detail_text.tag_configure("rule", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.tag_configure("num", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.tag_configure("example", foreground=p["RUBRIC"], font=th.FONT)
    detail_text.tag_configure("hint", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.config(state="disabled")
    themed.add(detail_text, bg="PANEL", fg="FG", insertbackground="FG")

    btn_row = tk.Frame(detail_frame, bg=p["PANEL"])
    btn_row.pack(fill="x", padx=14, pady=(0, 10))
    themed.add(btn_row, bg="PANEL")
    know_btn = ttk.Button(
        btn_row, text="✅ I KNOW IT", command=lambda: on_mark("known"), state="disabled",
    )
    know_btn.pack(side="left")
    learning_btn = ttk.Button(
        btn_row, text="\U0001F4D6 STILL LEARNING", command=lambda: on_mark("learning"), state="disabled",
    )
    learning_btn.pack(side="left", padx=(8, 0))

    def _on_select(_event):
        sel = tree.selection()
        on_select_rule(sel[0] if sel else None)

    tree.bind("<<TreeviewSelect>>", _on_select)

    return frame, {
        "tree": tree, "detail_header_var": detail_header_var, "detail_text": detail_text,
        "know_btn": know_btn, "learning_btn": learning_btn,
    }


class GrammarQuizWindow(tk.Toplevel):
    """A simpler sibling to study.StudyWindow, for grammar structures
    instead of vocabulary words, with two modes:

    - "structure" (default): shows a rule's NAME, reveal shows its
      definition/rule/example sentences — direct recall, like a
      vocabulary flashcard.
    - "identify": shows a SENTENCE from the book, reveal shows every
      grammar structure that sentence actually uses (there can be more
      than one — see build_sentence_index()) — recognizing grammar in
      context, the reverse direction from "structure" mode.

    No writing mode (StudyWindow's typed-answer-against-one-exact-word
    grading, and masking that word out of an example sentence, have no
    equivalent for a grammar rule or a whole illustrative sentence), and
    no SPEAK button (pronunciation doesn't apply to either a rule name
    or a full sentence). Everything else generic — due-first ordering,
    SM-2 scheduling via review_word()/mark_known(), the
    on_finish(known_ids, learning_ids) contract, the summary screen —
    mirrors StudyWindow exactly, reusing the same widget/layout choices
    (readonly Entry + Text for real copy/select, three button rows to
    avoid the narrow-window clipping StudyWindow's own history already
    ran into) rather than inventing new ones.

    `cards` is always [(key, payload), ...]: in "structure" mode `key`
    is a rule_id and `payload` is that rule's data dict; in "identify"
    mode `key` is a sentence and `payload` is the list of (rule_id,
    data) tuples build_sentence_index() found for it — the caller
    (app.py's open_grammar_quiz()) builds whichever shape matches
    `mode`, the same division of responsibility every other window here
    uses (the window renders what it's handed, the app decides what
    that is).
    """

    def __init__(self, master, cards, on_finish=None, palette=None, schedule=None,
                 already_known_rules=None, learned_at=None, mode="structure"):
        super().__init__(master)
        self.p = palette or th.LIGHT
        self.mode = mode if mode in ("structure", "identify") else "structure"
        self.title("Grammar Quiz — Text Analyzer")
        self.geometry("640x520")
        self.minsize(560, 420)
        self.configure(bg=self.p["BG"])
        self.on_finish = on_finish
        self.schedule = schedule
        self.learned_at = learned_at
        self.already_known_rules = already_known_rules if already_known_rules is not None else set()

        def _rule_ids_of(card):
            key, payload = card
            return [key] if self.mode == "structure" else [rid for rid, _d in payload]

        cards = list(cards)
        if self.schedule is not None:
            due_rule_ids = set(due_words(self.schedule, sorted({
                rid for card in cards for rid in _rule_ids_of(card)
            })))
            due_entries = [c for c in cards if any(rid in due_rule_ids for rid in _rule_ids_of(c))]
            later_entries = [c for c in cards if c not in due_entries]
        else:
            due_entries, later_entries = cards, []
        random.shuffle(due_entries)
        random.shuffle(later_entries)
        self.cards = due_entries + later_entries
        self._rule_ids_of = _rule_ids_of
        self.index = 0
        self.known_rules = []      # [(rule_id, name), ...] — deduped in show_summary()
        self.learning_rules = []   # [(rule_id, name), ...]
        self.revealed = False
        self.finished = False

        self.counter_label = tk.Label(self, text="", bg=self.p["BG"], fg=self.p["DIM"], font=th.FONT)
        self.counter_label.pack(pady=(20, 0))

        self.card_frame = tk.Frame(
            self, bg=self.p["PANEL"], highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"],
        )
        self.card_frame.pack(fill="both", expand=True, padx=30, pady=20)

        self.name_var = tk.StringVar()
        self.name_entry = tk.Entry(
            self.card_frame, textvariable=self.name_var, state="readonly",
            readonlybackground=self.p["PANEL"], fg=self.p["ACCENT"], font=th.FONT_CARD_WORD,
            relief="flat", highlightthickness=0, bd=0, justify="center", cursor="xterm",
            insertbackground=self.p["ACCENT"], selectbackground=self.p["ACCENT"],
            selectforeground=self.p["ACCENT_TEXT"],
        )

        self.card_text = tk.Text(
            self.card_frame, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word",
            relief="flat", highlightthickness=0, padx=0, pady=0, borderwidth=0,
            cursor="xterm", insertbackground=self.p["FG"], height=9,
        )
        self.card_text.tag_configure("def", foreground=self.p["FG"], font=th.FONT, justify="center")
        self.card_text.tag_configure("rule", foreground=self.p["DIM"], font=th.FONT_SMALL_ITALIC, justify="center")
        self.card_text.tag_configure("example", foreground=self.p["RUBRIC"], font=th.FONT_SMALL_ITALIC, justify="center")
        # "identify" mode's prompt (the sentence itself, shown before
        # reveal, in place of "structure" mode's name_entry) and the
        # bold rule-name lead-in each reveal() bullet starts with.
        self.card_text.tag_configure("prompt", foreground=self.p["ACCENT"], font=th.FONT_ROW_WORD, justify="center")
        self.card_text.tag_configure("answer_name", foreground=self.p["ACCENT"], font=th.FONT_BOLD, justify="center")
        self.card_text.config(state="disabled")

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

        self.reveal_btn = ttk.Button(
            self, text="SHOW ANSWER (space)", command=self.reveal, style="Accent.TButton",
        )
        self.reveal_btn.pack(fill="x", padx=30, pady=(0, 8))

        self.verdict_row = tk.Frame(self, bg=self.p["BG"])
        self.learning_btn = ttk.Button(
            self.verdict_row, text="STILL LEARNING (2)", command=lambda: self.next_card(False),
        )
        self.learning_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.known_btn = ttk.Button(
            self.verdict_row, text="I KNOW IT (1)", command=lambda: self.next_card(True), style="Accent.TButton",
        )
        self.known_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.verdict_row.pack(fill="x", padx=30, pady=(0, 8))

        utility_row = tk.Frame(self, bg=self.p["BG"])
        utility_row.pack(fill="x", padx=30, pady=(0, 20))
        self.end_btn = ttk.Button(utility_row, text="END SESSION", command=self.end_session)
        self.end_btn.pack(side="right")

        self.bind("<space>", lambda _e: self.reveal())
        self.bind("<Return>", lambda _e: self.next_card(True))
        self.bind("<Key-1>", lambda _e: self.next_card(True))
        self.bind("<Key-2>", lambda _e: self.next_card(False))
        th.apply_dark_titlebar(self, self.p is th.DARK)
        self.show_card()
        self.focus_set()

    def show_card(self):
        # A card is skipped once every rule it involves is already known
        # — for a "structure" card that's just its one rule_id; for an
        # "identify" card (possibly several rules at once), ALL of them
        # need to already be known before the sentence itself stops
        # being worth asking about.
        while self.index < len(self.cards) and all(
            rid in self.already_known_rules for rid in self._rule_ids_of(self.cards[self.index])
        ):
            del self.cards[self.index]
        if self.index >= len(self.cards):
            self.show_summary()
            return
        self.summary_frame.pack_forget()
        self.card_text.pack(pady=10, padx=20, fill="x")
        key, _payload = self.cards[self.index]
        self.card_text.config(state="normal")
        self.card_text.delete("1.0", "end")
        if self.mode == "identify":
            # The sentence itself IS the prompt — shown up front, unlike
            # "structure" mode where the name is the starting point and
            # the sentence(s) only appear after reveal().
            self.name_entry.pack_forget()
            self.card_text.insert("end", f"“{key}”", "prompt")
            self.name_var.set("")
        else:
            self.name_entry.pack(pady=(40, 10))
            self.name_var.set(key)
        self.card_text.config(state="disabled")
        self.revealed = False
        self.reveal_btn.config(state="normal")
        self.counter_label.config(text=f"{self.index + 1}/{len(self.cards)}")

    def reveal(self):
        if self.finished or self.revealed:
            return
        _key, payload = self.cards[self.index]
        self.card_text.config(state="normal")
        if self.mode == "identify":
            # payload is [(rule_id, data), ...] — could be more than one
            # real answer for the same sentence (see
            # build_sentence_index()'s docstring), so every match is
            # listed, not just the first.
            self.card_text.insert("end", "\n\nThis sentence uses:\n", "rule")
            for _rid, data in payload:
                self.card_text.insert("end", f"\n{data['name']}", "answer_name")
                self.card_text.insert("end", f" — {data['definition']}", "def")
        else:
            # payload is a single rule's data dict.
            self.card_text.insert("end", payload["definition"], "def")
            self.card_text.insert("end", "\n\n" + payload["rule_explanation"], "rule")
            # Up to 3 illustrative sentences, not the full (possibly
            # hundreds-long) list — a quiz card is meant to jog recall,
            # not reproduce the whole detail-panel listing.
            for sentence in payload["sentences"][:3]:
                self.card_text.insert("end", f"\n\n“{sentence}”", "example")
        self.card_text.config(state="disabled")
        self.revealed = True

    def next_card(self, known):
        if self.finished:
            return
        for rule_id, name in self._card_rule_names(self.index):
            (self.known_rules if known else self.learning_rules).append((rule_id, name))
            if self.schedule is not None:
                review_word(self.schedule, rule_id, known)
            if known and self.learned_at is not None:
                mark_known(self.learned_at, rule_id)
        self.index += 1
        self.show_card()

    def _card_rule_names(self, index):
        """[(rule_id, name), ...] for cards[index] — one entry for a
        "structure" card, possibly several for an "identify" card whose
        sentence matched more than one rule. Marking such a card known
        marks every one of its rules known: recognizing all the grammar
        in a sentence is the whole point of that mode's verdict.
        """
        key, payload = self.cards[index]
        if self.mode == "identify":
            return [(rid, data["name"]) for rid, data in payload]
        return [(key, payload["name"])]

    def end_session(self):
        if not self.finished:
            self.show_summary()

    def show_summary(self):
        self.finished = True
        self.name_entry.pack_forget()
        self.card_text.pack_forget()
        self.summary_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Dedupe by rule_id: "identify" mode can mark the SAME rule
        # known/learning more than once (it appeared in several
        # sentence-cards this session) — a rule marked known on any card
        # wins over a "still learning" verdict from a different card in
        # the same session, same "known wins" precedent
        # app.py's _record_grammar_quiz_results() already applies when
        # merging a session's results into long-term progress.
        names_by_id = dict(self.known_rules) | dict(self.learning_rules)
        known_ids = {rid for rid, _name in self.known_rules}
        learning_ids = {rid for rid, _name in self.learning_rules} - known_ids

        if self.on_finish:
            self.on_finish(sorted(known_ids), sorted(learning_ids))

        reviewed = len(self.known_rules) + len(self.learning_rules)
        not_reached = len(self.cards) - reviewed
        self.counter_label.config(text=f"{reviewed}/{len(self.cards)} reviewed")
        self.reveal_btn.config(state="disabled")
        self.learning_btn.config(state="disabled")
        self.known_btn.config(state="disabled")
        self.end_btn.config(text="CLOSE", command=self.destroy)

        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        header = f"SESSION SUMMARY — {reviewed} reviewed"
        if not_reached:
            header += f", {not_reached} not reached"
        self.summary_text.insert("end", header + "\n\n", "heading")

        self.summary_text.insert("end", f"I KNOW IT  ({len(known_ids)})\n", "heading")
        if known_ids:
            for rid in sorted(known_ids, key=lambda r: names_by_id[r]):
                self.summary_text.insert("end", f"  {names_by_id[rid]}\n", "known")
        else:
            self.summary_text.insert("end", "  (none)\n", "known")

        self.summary_text.insert("end", "\n")
        self.summary_text.insert("end", f"STILL LEARNING  ({len(learning_ids)})\n", "heading")
        if learning_ids:
            for rid in sorted(learning_ids, key=lambda r: names_by_id[r]):
                self.summary_text.insert("end", f"  {names_by_id[rid]}\n", "learning")
        else:
            self.summary_text.insert("end", "  (none)\n", "learning")
        self.summary_text.config(state="disabled")


def build_grammar_readiness_card(parent, palette, on_start, on_export_anki, themed=None):
    """The docked "grammar readiness" card for the main window's right
    column when Section 2 is active — mirrors
    study.build_study_summary_card()'s shape and wording, two buttons
    into GrammarQuizWindow (one per `mode`, same "one button per mode"
    pattern build_study_summary_card()'s flashcard/writing pair already
    uses) plus one into Anki export.

    `on_start` is called with a single "structure"/"identify" argument
    (which GrammarQuizWindow mode to open), mirroring
    build_study_summary_card()'s own `on_start("flashcard"/"writing")`
    contract. `on_export_anki` takes none, same as its vocab sibling.

    Returns (frame, widgets) — widgets: book_label, meta_label,
    readiness_label, progress_bar, known_label, to_learn_label,
    count_label, quiz_btn, identify_btn, anki_btn.
    """
    p = palette
    frame = tk.Frame(parent, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
    header = tk.Label(frame, text="GRAMMAR READINESS", bg=p["PANEL"], fg=p["FG"], font=th.FONT_BOLD)
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
        frame, text="0 structure(s) ready to study", bg=p["PANEL"], fg=p["DIM"], font=th.FONT,
    )
    count_label.pack(anchor="w", padx=12, pady=(0, 6))
    quiz_btn = ttk.Button(
        frame, text="STRUCTURE CARDS", command=lambda: on_start("structure"), style="Accent.TButton",
    )
    quiz_btn.pack(padx=12, pady=(0, 6), fill="x")
    # Shortened from "IDENTIFY IN SENTENCE"/"EXPORT GRAMMAR TO ANKI" —
    # confirmed directly (screenshot) that the longer text overflowed
    # past the button's own edge, since ttk.Button neither wraps nor
    # shrinks its label to fit.
    identify_btn = ttk.Button(
        frame, text="\U0001F50D IDENTIFY GRAMMAR", command=lambda: on_start("identify"),
    )
    identify_btn.pack(padx=12, pady=(0, 6), fill="x")
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
        "count_label": count_label, "quiz_btn": quiz_btn, "identify_btn": identify_btn, "anki_btn": anki_btn,
    }
    return frame, widgets
