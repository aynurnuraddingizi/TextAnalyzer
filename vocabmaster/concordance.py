"""Concordance / "Find in Book" window: every sentence in the loaded
book containing a searched word or phrase, in reading order, with the
match itself highlighted — what a close reader or researcher needs when
one example sentence (all the Detail panel's "example" ever shows, and
only for words already in the glossary) isn't enough to see how a term
is actually used across a whole text. Free-text: not limited to
glossary words — proper nouns, phrases, motifs all work too.
"""
import tkinter as tk
from tkinter import ttk

from text_analyzer import find_occurrences

import theme as th

MAX_RESULTS = 300


class ConcordanceWindow(tk.Toplevel):
    def __init__(self, master, sentences, palette, initial_query=""):
        super().__init__(master)
        self.p = palette
        self.sentences = sentences
        self.title("Find in Book")
        self.geometry("720x560")
        self.minsize(480, 360)
        self.configure(bg=self.p["BG"])

        top = tk.Frame(self, bg=self.p["BG"])
        top.pack(fill="x", padx=20, pady=(16, 8))
        tk.Label(top, text="FIND IN BOOK", bg=self.p["BG"], fg=self.p["FG"], font=th.FONT_BOLD).pack(anchor="w")
        tk.Label(
            top, text="Every sentence containing a word or phrase, in reading order.",
            bg=self.p["BG"], fg=self.p["DIM"], font=th.FONT,
        ).pack(anchor="w", pady=(0, 8))

        search_row = tk.Frame(top, bg=self.p["BG"])
        search_row.pack(fill="x")
        self.query_var = tk.StringVar(value=initial_query)
        self.entry = tk.Entry(
            search_row, textvariable=self.query_var, bg=self.p["PANEL"], fg=self.p["FG"],
            insertbackground=self.p["FG"], font=th.FONT, relief="flat",
            highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"], highlightcolor=self.p["ACCENT"],
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", lambda _e: self.search())
        self.search_btn = ttk.Button(search_row, text="SEARCH", command=self.search, style="Accent.TButton")
        self.search_btn.pack(side="left", padx=(8, 0))

        opts_row = tk.Frame(top, bg=self.p["BG"])
        opts_row.pack(fill="x", pady=(8, 0))
        self.whole_word_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts_row, text="Whole word only", variable=self.whole_word_var, command=self.search,
        ).pack(side="left")
        self.copy_btn = ttk.Button(opts_row, text="COPY ALL", command=self.copy_all)
        self.copy_btn.pack(side="left", padx=(12, 0))
        self.count_label = tk.Label(opts_row, text="", bg=self.p["BG"], fg=self.p["DIM"], font=th.FONT_SMALL_ITALIC)
        self.count_label.pack(side="right")

        body = tk.Frame(self, bg=self.p["PANEL"], highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"])
        body.pack(fill="both", expand=True, padx=20, pady=(8, 20))
        scrollbar = tk.Scrollbar(
            body, bg=self.p["PANEL"], troughcolor=self.p["BG"], activebackground=self.p["SECOND_HOVER"],
            highlightthickness=0,
        )
        scrollbar.pack(side="right", fill="y")
        self.result_text = tk.Text(
            body, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word", relief="flat",
            yscrollcommand=scrollbar.set, highlightthickness=0, padx=12, pady=10, cursor="xterm",
            insertbackground=self.p["FG"],
        )
        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)
        self.result_text.tag_configure("match", foreground=self.p["ACCENT"], font=th.FONT_BOLD)
        self.result_text.tag_configure("num", foreground=self.p["DIM"], font=th.FONT_SMALL_ITALIC)
        self.result_text.tag_configure("dim", foreground=self.p["DIM"], font=th.FONT_SMALL_ITALIC)
        self.result_text.config(state="disabled")

        self.bind("<Escape>", lambda _e: self.destroy())
        th.apply_dark_titlebar(self, self.p is th.DARK)
        self.entry.focus_set()
        self.entry.icursor("end")
        if initial_query.strip():
            self.after(50, self.search)

    def search(self):
        query = self.query_var.get().strip()
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        if not query:
            self.result_text.config(state="disabled")
            self.count_label.config(text="")
            return
        matches, total = find_occurrences(
            self.sentences, query, whole_word=self.whole_word_var.get(), max_results=MAX_RESULTS,
        )
        if not matches:
            self.result_text.insert("end", "No matches found in this book.", "dim")
        for i, (sentence, spans) in enumerate(matches, start=1):
            self.result_text.insert("end", f"{i}.  ", "num")
            pos = 0
            for start, end in spans:
                self.result_text.insert("end", sentence[pos:start])
                self.result_text.insert("end", sentence[start:end], "match")
                pos = end
            self.result_text.insert("end", sentence[pos:] + "\n\n")
        self.result_text.config(state="disabled")
        if total == 0:
            self.count_label.config(text="No matches")
        elif total > len(matches):
            self.count_label.config(text=f"Showing first {len(matches)} of {total} matches")
        else:
            self.count_label.config(text=f"{total} match" if total == 1 else f"{total} matches")

    def copy_all(self):
        content = self.result_text.get("1.0", "end").strip()
        if not content:
            return
        self.clipboard_clear()
        self.clipboard_append(content)
