"""First-launch onboarding wizard: a short, skippable walkthrough of the
app's main features. Shown once automatically (gated on
settings["onboarded"], persisted via save_settings) and replayable any
time from the topbar's tutorial button.
"""
import tkinter as tk
from tkinter import ttk

import theme as th

PAGES = [
    (
        "Welcome to Text Analyzer",
        "Pick any PDF or EPUB and this app extracts every unique word in "
        "it, then builds an instant, fully offline glossary — no internet "
        "needed except for the optional pronunciation feature.",
    ),
    (
        "1. Select a book",
        "Click SELECT BOOK (or a title under RECENT BOOKS in the "
        "sidebar), choose the book's language or leave it on Auto-detect, "
        "then click GENERATE GLOSSARY. A progress bar tracks the run, and "
        "STOP cancels it early if needed.",
    ),
    (
        "2. Browse & inspect words",
        "The word table lists every word found, with its count and a "
        "short definition. Type in FILTER to narrow it down, click a "
        "column header area to sort by frequency or A-Z, and click a row "
        "to see its full definition, a real example sentence from the "
        "book, and a SPEAK button in the Detail panel below. Click "
        "several rows to select many words at once.",
    ),
    (
        "3. Track what you know",
        "Mark any word I KNOW IT or STILL LEARNING (works on a "
        "multi-selection too) — this is remembered across every book you "
        "open. Turn on \"Hide words I already know\" to keep the list "
        "focused on what's actually new.",
    ),
    (
        "4. Study mode",
        "In the right-hand column, FLASHCARDS shows the word and you "
        "self-report whether you knew it (Space to reveal, 1/2, S to "
        "hear it spoken). WRITING QUIZ flips it around: you see the "
        "definition and an example with the word blanked out, and type "
        "the word yourself — get it right or wrong and it's graded "
        "automatically, no self-reporting involved.",
    ),
    (
        "5. Export your glossary",
        "SAVE AS... writes the glossary as TXT, DOCX, PDF, CSV, JSON, or "
        "Markdown. EXPORT TO ANKI writes a ready-to-import flashcard deck "
        "for Anki's own text importer — just open the file from Anki's "
        "File > Import.",
    ),
    (
        "6. Find in Book",
        "One example sentence isn't always enough — click FIND IN BOOK "
        "(or FIND ALL next to a selected word) to see every sentence a "
        "word or phrase appears in, in reading order. Works for names and "
        "phrases too, not just dictionary words — useful for tracing a "
        "character, theme, or term through a whole book.",
    ),
    (
        "You're set",
        "Your theme choice, recent books, and known/learning progress are "
        "all saved automatically. Replay this tour anytime from the 🎓 "
        "button in the top-right corner.",
    ),
]


class OnboardingWizard(tk.Toplevel):
    def __init__(self, master, palette, on_done=None):
        super().__init__(master)
        self.p = palette
        self.on_done = on_done
        self.index = 0
        self.title("Welcome")
        self.geometry("520x360")
        self.minsize(460, 320)
        self.configure(bg=self.p["BG"])
        self.transient(master)
        self.resizable(True, True)

        self.step_label = tk.Label(self, text="", bg=self.p["BG"], fg=self.p["DIM"], font=th.FONT_SMALL_BOLD)
        self.step_label.pack(anchor="w", padx=30, pady=(24, 0))

        # FONT_TITLE (used for the main window's own heading), not the
        # larger FONT_CARD_WORD sized for StudyWindow's single flashcard
        # word — that, combined with no wraplength, forced this window
        # far wider than its explicit geometry on a scaled-DPI display,
        # pushing most of the dialog off-screen. wraplength is still set
        # as a second line of defense in case a future page title is long.
        self.title_label = tk.Label(
            self, text="", bg=self.p["BG"], fg=self.p["ACCENT"], font=th.FONT_TITLE,
            wraplength=460, justify="left",
        )
        self.title_label.pack(anchor="w", padx=30, pady=(4, 14), fill="x")

        self.body_label = tk.Label(
            self, text="", bg=self.p["BG"], fg=self.p["FG"], font=th.FONT, wraplength=460, justify="left",
        )
        self.body_label.pack(anchor="w", padx=30, fill="both", expand=True)

        nav = tk.Frame(self, bg=self.p["BG"])
        nav.pack(fill="x", padx=30, pady=20)
        self.skip_btn = ttk.Button(nav, text="SKIP", command=self._finish)
        self.skip_btn.pack(side="left")
        self.back_btn = ttk.Button(nav, text="BACK", command=self._back)
        self.back_btn.pack(side="right", padx=(8, 0))
        self.next_btn = ttk.Button(nav, text="NEXT", command=self._next, style="Accent.TButton")
        self.next_btn.pack(side="right")

        th.apply_dark_titlebar(self, self.p is th.DARK)
        self._show_page()
        self.grab_set()
        self.focus_set()

    def _show_page(self):
        title, body = PAGES[self.index]
        self.step_label.config(text=f"{self.index + 1} / {len(PAGES)}")
        self.title_label.config(text=title)
        self.body_label.config(text=body)
        self.back_btn.config(state="disabled" if self.index == 0 else "normal")
        last = self.index == len(PAGES) - 1
        self.next_btn.config(text="GET STARTED" if last else "NEXT")

    def _next(self):
        if self.index == len(PAGES) - 1:
            self._finish()
            return
        self.index += 1
        self._show_page()

    def _back(self):
        if self.index > 0:
            self.index -= 1
            self._show_page()

    def _finish(self):
        if self.on_done:
            self.on_done()
        self.destroy()
