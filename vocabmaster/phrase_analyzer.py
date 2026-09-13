"""Section 3: Phraseology — two clearly separate tiers, matching
extract_phrases()'s own "recognized"/"recurring" split (see its
docstring in text_analyzer.py for the full reasoning):

  - RECOGNIZED PHRASES & IDIOMS: real, editorially-catalogued idioms,
    phrasal verbs, and fixed expressions (from the bundled Wiktionary-
    derived phrase_en.json/phrase_es.json — English and Spanish books
    only, see extract_phrases()'s docstring) actually found in this
    book — split into "Idioms" (Wiktionary's own "idiomatic" tag) and
    "Phrasal Verbs & Fixed Expressions" (everything else recognized) so
    the two very different kinds of "this is real" aren't run together.
  - OTHER RECURRING COMBINATIONS: the original statistical approach —
    word sequences that repeat often in THIS book but aren't in the
    dictionary — grouped by length as before.

Deliberately no know/learning tracking or quiz mode here, unlike
Vocabulary and Grammar: a RECURRING phrase found this way is a
statistical artifact of THIS book (there's no fixed, book-independent
catalog of "recurring combinations to learn" the way there's a fixed
set of grammar structures or CEFR-leveled words). RECOGNIZED phrases
arguably could support tracking (they ARE a fixed catalog, same as
grammar structures) — left out of v1 for scope, a natural follow-on if
wanted. Browsing and export are the whole feature for now.
"""
import tkinter as tk
from tkinter import ttk

import theme as th

_LENGTH_LABELS = {2: "2-word", 3: "3-word", 4: "4-word", 5: "5-word"}
_RECOGNIZED_IDIOM_CAT = "recognized_idiom"
_RECOGNIZED_OTHER_CAT = "recognized_other"


def build_phrase_section(parent, palette, themed, on_select_phrase, search_var):
    """Builds the category/phrase Treeview on the left, a scrollable
    detail panel (stats + every matching example this book actually
    has) on the right — same left/right, real-vertical-room-for-many-
    sentences layout build_grammar_section() uses, for the same reason.

    `on_select_phrase` is called with the selected phrase's tree iid (or
    None when the selection clears) — app.py owns self.phrases and
    populates the detail widgets itself, same division of responsibility
    as the vocabulary/grammar trees. See populate_phrase_tree() for the
    iid scheme app.py needs to parse this back out of.

    `search_var` is app.py's self.phrase_search_var (a tk.StringVar it
    already owns and traces — same "app.py creates the Var, this
    function just wires an Entry to it" split as the rest of this app's
    search boxes) — filters the tree live as it's typed into, same
    "FILTER" behavior the Vocabulary section already has (there was no
    way to find one specific phrase in a long list otherwise, once a
    book turns up more than a screenful of recognized/recurring
    phrases).

    Returns (frame, widgets) — widgets = {"tree", "search_entry",
    "detail_header_var", "detail_text"}.
    """
    p = palette
    frame = tk.Frame(parent, bg=p["BG"])

    # 400/320, not something narrower: confirmed directly (screenshot)
    # that the section headings below ("RECOGNIZED PHRASES & IDIOMS")
    # got cut off mid-word at anything less — same class of bug, and
    # same fix, as build_grammar_section()'s own category-row width.
    list_frame = tk.Frame(frame, bg=p["BG"], width=400)
    list_frame.pack(side="left", fill="y", padx=(0, 10))
    list_frame.pack_propagate(False)
    themed.add(list_frame, bg="BG")

    header = tk.Label(
        list_frame, text="PHRASES FOUND", bg=p["BG"], fg=p["FG"], font=th.FONT_BOLD,
        wraplength=380, justify="left", anchor="w",
    )
    header.pack(anchor="w", fill="x", pady=(0, 6))
    themed.add(header, bg="BG", fg="FG")

    search_frame = tk.Frame(list_frame, bg=p["BG"])
    search_frame.pack(fill="x", pady=(0, 8))
    themed.add(search_frame, bg="BG")
    search_hdr = tk.Label(search_frame, text="SEARCH", bg=p["BG"], fg=p["FG"], font=th.FONT_SMALL_BOLD)
    search_hdr.pack(anchor="w")
    themed.add(search_hdr, bg="BG", fg="FG")
    search_entry = tk.Entry(
        search_frame, textvariable=search_var, bg=p["PANEL"], fg=p["FG"], insertbackground=p["FG"],
        font=th.FONT, relief="flat", highlightthickness=1, highlightbackground=p["PANEL_BORDER"],
        highlightcolor=p["ACCENT"],
    )
    search_entry.pack(fill="x", pady=(2, 0), ipady=4)
    themed.add(
        search_entry, bg="PANEL", fg="FG", insertbackground="FG",
        highlightbackground="PANEL_BORDER", highlightcolor="ACCENT",
    )

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
    tree.heading("#0", text="Phrase")
    tree.heading("count", text="Count")
    tree.column("#0", width=320, anchor="w", stretch=False)
    tree.column("count", width=60, anchor="center", stretch=False)
    tree.tag_configure("section", foreground=p["ACCENT"], font=th.FONT_BOLD)
    tree.tag_configure("category", foreground=p["ACCENT"], font=th.FONT_SMALL_BOLD)
    tree.tag_configure("idiomatic", font=th.FONT_ROW_WORD)

    detail_frame = tk.Frame(frame, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
    detail_frame.pack(side="left", fill="both", expand=True)
    themed.add(detail_frame, bg="PANEL", highlightbackground="PANEL_BORDER")

    detail_header_var = tk.StringVar(value="Select a phrase to see where it's used")
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
    detail_text.tag_configure("stat", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.tag_configure("num", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.tag_configure("example", foreground=p["FG"], font=th.FONT)
    detail_text.tag_configure("hint", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.config(state="disabled")
    themed.add(detail_text, bg="PANEL", fg="FG", insertbackground="FG")

    tree.bind("<<TreeviewSelect>>", lambda _e: on_select_phrase(_selected_key(tree)))

    widgets = {
        "tree": tree, "search_entry": search_entry,
        "detail_header_var": detail_header_var, "detail_text": detail_text,
    }
    return frame, widgets


def _selected_key(tree):
    sel = tree.selection()
    if not sel or not sel[0].startswith(("recognized::", "recurring::")):
        return None
    return sel[0]


def populate_phrase_tree(tree, phrases, query=""):
    """(Re)builds `tree` from `phrases` — extract_phrases()'s own
    {"recognized": [...], "recurring": {length: {...}}} return shape.

    iid scheme (parsed back out by app.py's _phrase_entry_by_iid()):
      - "recognized::<phrase text>" — a recognized-tier entry; the
        phrase text itself is the key into the flat "recognized" list
        (each canonical phrase appears once, so this is unambiguous).
      - "recurring::<length>::<phrase text>" — a recurring-tier entry;
        needs the length prefix since "recurring" is grouped by length
        and, unlike "recognized", isn't keyed uniquely by text alone.
      - "section::..."/"category::..." — header rows, never selectable
        (see _selected_key()).

    Two top-level SECTION rows ("RECOGNIZED PHRASES & IDIOMS" /
    "OTHER RECURRING COMBINATIONS"), each auto-expanded; under the
    first, two CATEGORY rows ("Idioms" / "Phrasal Verbs & Fixed
    Expressions") splitting on Wiktionary's own "idiomatic" tag; under
    the second, one CATEGORY row per length, exactly as before. A
    section/category with nothing in it is simply omitted, same
    "don't show what wasn't there" convention the rest of this app uses.

    `query`: a case-insensitive substring filter against each entry's
    own phrase text (app.py's self.phrase_search_var, already
    lower-cased/stripped there) — matches Vocabulary's own FILTER box.
    A section/category left empty by the filter is omitted exactly the
    same way one left empty by the book itself already is, so "no
    matches" and "book has none of these" look identical, which is
    the right behavior — the user doesn't need to know which case
    they're looking at, only that there's nothing to show.
    """
    tree.delete(*tree.get_children())
    query = (query or "").strip().lower()

    def matches(entry):
        return not query or query in entry["phrase"].lower()

    recognized = [e for e in (phrases.get("recognized") or []) if matches(e)]
    if recognized:
        # Short labels, not "RECOGNIZED PHRASES & IDIOMS"/"Phrasal Verbs
        # & Fixed Expressions" — confirmed directly (screenshot) that
        # the longer wording got cut off mid-word at this column width.
        tree.insert(
            "", "end", iid="section::recognized", text="✅ RECOGNIZED PHRASES",
            values=("",), tags=("section",), open=True,
        )
        idioms = [e for e in recognized if e["idiomatic"]]
        others = [e for e in recognized if not e["idiomatic"]]
        for cat_key, label, entries in (
            (_RECOGNIZED_IDIOM_CAT, "Idioms", idioms),
            (_RECOGNIZED_OTHER_CAT, "Other Phrases", others),
        ):
            if not entries:
                continue
            cat_iid = f"category::{cat_key}"
            tree.insert(
                "section::recognized", "end", iid=cat_iid, text=f"▽ {label} ({len(entries)})",
                values=("",), tags=("category",), open=True,
            )
            for entry in entries:
                iid = f"recognized::{entry['phrase']}"
                tree.insert(
                    cat_iid, "end", iid=iid, text=entry["phrase"], values=(entry["count"],),
                    tags=("idiomatic",) if entry["idiomatic"] else (),
                )

    recurring = phrases.get("recurring") or {}
    filtered_recurring = {
        length: [e for e in data["entries"] if matches(e)]
        for length, data in recurring.items()
    }
    if any(filtered_recurring.values()):
        tree.insert(
            "", "end", iid="section::recurring", text="\U0001F4CA OTHER RECURRING COMBINATIONS",
            values=("",), tags=("section",), open=bool(not recognized),
        )
        for length in sorted(recurring.keys()):
            data = recurring[length]
            entries = filtered_recurring[length]
            if not entries:
                continue
            label = _LENGTH_LABELS.get(length, f"{length}-word") + " phrases"
            cat_iid = f"category::recurring::{length}"
            tree.insert(
                "section::recurring", "end", iid=cat_iid, text=f"▽ {label}", values=("",),
                tags=("category",), open=bool(not recognized),
            )
            for entry in entries:
                iid = f"recurring::{length}::{entry['phrase']}"
                tree.insert(cat_iid, "end", iid=iid, text=entry["phrase"], values=(entry["count"],))
            if not query and data["total_found"] > len(entries):
                tree.insert(
                    cat_iid, "end", iid=f"category::recurring::{length}::more", open=False,
                    text=f"  (showing top {len(entries)} of {data['total_found']})", values=("",),
                )
