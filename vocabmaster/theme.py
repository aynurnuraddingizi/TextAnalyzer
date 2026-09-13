"""Light/dark color palettes and the ttk style setup shared by every
window in the app.

Both palettes share one "bold coffee" brown accent family (the same
hue, adjusted in brightness per background so it stays readable): LIGHT
is white with dark-roast coffee accents/text, DARK is black with a
brighter caramel-coffee accent. Same color *roles* in both
(background/panel/accent/dim/rubric) so toggling themes doesn't change
which thing on screen means what, just how it looks.
"""
import ctypes
import sys

FONT = ("Georgia", 11)
FONT_BOLD = ("Georgia", 11, "bold")
FONT_TITLE = ("Georgia", 20, "bold")
FONT_SMALL_BOLD = ("Georgia", 10, "bold")
FONT_SMALL_ITALIC = ("Georgia", 10, "italic")
FONT_CARD_WORD = ("Georgia", 26, "bold")
FONT_ROW_WORD = ("Georgia", 13, "bold")

LIGHT = {
    "BG": "#ffffff",            # white page
    "PANEL": "#f7f2ec",         # faint warm-white panel (inputs, cards)
    "PANEL_BORDER": "#c9b6a0",  # light coffee-with-cream border
    "FG": "#2a1a10",            # bold dark-roast coffee (main text)
    "DIM": "#7a6455",           # muted coffee (secondary text)
    "ACCENT": "#4b2e1f",        # bold coffee (primary buttons)
    "ACCENT_HOVER": "#63402c",
    "ACCENT_TEXT": "#f7f0e8",   # cream text on accent buttons
    "SECOND_BG": "#ece1d3",     # light coffee-with-cream (secondary buttons)
    "SECOND_HOVER": "#ddcdb8",
    "SECOND_FG": "#3b2a1c",
    "DISABLED_BG": "#e9e0d5",
    "DISABLED_FG": "#ab9a8a",
    "RUBRIC": "#7a3b1e",        # cinnamon-coffee, used in examples/flashcards
}

DARK = {
    "BG": "#000000",            # black page
    "PANEL": "#161210",         # near-black warm panel (inputs, cards)
    "PANEL_BORDER": "#4a3626",  # dark coffee border
    "FG": "#f0e6da",            # cream (main text)
    "DIM": "#b39c85",           # muted tan (secondary text)
    "ACCENT": "#a3703f",        # bold coffee, brightened for visibility on black
    "ACCENT_HOVER": "#bd8752",
    "ACCENT_TEXT": "#100b08",   # near-black text on accent buttons
    "SECOND_BG": "#241c15",     # dark coffee-bean (secondary buttons)
    "SECOND_HOVER": "#362a1c",
    "SECOND_FG": "#f0e6da",
    "DISABLED_BG": "#1a1512",
    "DISABLED_FG": "#5c4e3f",
    "RUBRIC": "#d19461",        # warm coffee-caramel, used in examples/flashcards
}

PALETTES = {"light": LIGHT, "dark": DARK}


def apply_dark_titlebar(window, dark):
    """Best-effort: toggle the native Windows title bar to match the
    app's own theme. Tkinter can't style OS window chrome itself, so
    without this the title bar stayed light-gray regardless of theme —
    confirmed directly, the one piece of "the whole page" that dark mode
    never actually reached. Silently does nothing on non-Windows, on
    Windows versions that don't support it, or if it fails for any other
    reason — purely cosmetic, never worth crashing over.
    """
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        # DWMWA_USE_IMMERSIVE_DARK_MODE is 20 on Windows 11 / 10 20H1+,
        # 19 on some earlier builds that still support it — try both,
        # there's no cheap way to know ahead of time which one a given
        # build wants, and setting the "wrong" one is a harmless no-op.
        for attr in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)
            )
    except Exception:
        pass


def build_ttk_style(style, palette):
    """(Re)configure every ttk style this app uses from `palette`. Safe
    to call again after a theme switch — ttk.Style().configure just
    overwrites the previous values for each style name.
    """
    p = palette
    style.theme_use("clam")

    style.configure(
        "TButton",
        background=p["SECOND_BG"], foreground=p["SECOND_FG"], bordercolor=p["PANEL_BORDER"],
        borderwidth=1, focusthickness=0, focuscolor=p["SECOND_BG"], padding=(14, 8), font=FONT_BOLD,
    )
    style.map(
        "TButton",
        background=[("active", p["SECOND_HOVER"]), ("disabled", p["DISABLED_BG"])],
        foreground=[("disabled", p["DISABLED_FG"])],
        bordercolor=[("disabled", p["DISABLED_BG"])],
        # "clam" draws its dotted focus indicator in a fixed color that
        # ignores the palette entirely — confirmed directly, STOP (which
        # had keyboard focus at launch) rendered as a stray bright-white
        # box in dark mode even though every other disabled button was
        # correctly dark. Pointing focuscolor at the button's own
        # background for every state makes that indicator blend in
        # instead of standing out as an unthemed patch.
        focuscolor=[("disabled", p["DISABLED_BG"]), ("!disabled", p["SECOND_BG"])],
    )

    style.configure(
        "Accent.TButton",
        background=p["ACCENT"], foreground=p["ACCENT_TEXT"], bordercolor=p["ACCENT"],
        borderwidth=0, focusthickness=0, focuscolor=p["ACCENT"], padding=(16, 9), font=FONT_BOLD,
    )
    style.map(
        "Accent.TButton",
        background=[("active", p["ACCENT_HOVER"]), ("disabled", p["DISABLED_BG"])],
        foreground=[("disabled", p["DISABLED_FG"])],
        focuscolor=[("disabled", p["DISABLED_BG"]), ("!disabled", p["ACCENT"])],
    )

    # Small, quiet icon-style buttons for the topbar (dark-mode toggle,
    # settings, help) — same secondary coloring, less padding.
    style.configure(
        "Icon.TButton",
        background=p["BG"], foreground=p["FG"], bordercolor=p["BG"],
        borderwidth=0, focusthickness=0, focuscolor=p["BG"], padding=(8, 4), font=FONT_BOLD,
    )
    style.map(
        "Icon.TButton",
        background=[("active", p["SECOND_BG"])],
    )

    style.configure(
        "TProgressbar",
        background=p["ACCENT"], troughcolor=p["PANEL"], bordercolor=p["PANEL_BORDER"],
        lightcolor=p["ACCENT"], darkcolor=p["ACCENT"],
    )
    style.configure("TCheckbutton", background=p["BG"], foreground=p["FG"], font=FONT)
    style.map(
        "TCheckbutton",
        background=[("active", p["BG"])],
        indicatorcolor=[("selected", p["ACCENT"])],
    )
    style.configure("TRadiobutton", background=p["BG"], foreground=p["FG"], font=FONT)
    style.map(
        "TRadiobutton",
        background=[("active", p["BG"])],
        indicatorcolor=[("selected", p["ACCENT"])],
    )

    style.configure(
        "TCombobox",
        fieldbackground=p["PANEL"], background=p["SECOND_BG"], foreground=p["FG"],
        arrowcolor=p["FG"], bordercolor=p["PANEL_BORDER"], font=FONT,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", p["PANEL"])],
        foreground=[("readonly", p["FG"])],
        background=[("active", p["SECOND_HOVER"])],
    )

    style.configure(
        "Treeview",
        background=p["PANEL"], fieldbackground=p["PANEL"], foreground=p["FG"],
        bordercolor=p["PANEL_BORDER"], borderwidth=1, font=FONT, rowheight=26,
    )
    # NOT p["ACCENT"] for the selected background: row tags ("new" is
    # colored ACCENT-on-purpose, to highlight undecided words) win over
    # ttk's selected-state foreground in Treeview — confirmed directly,
    # selecting an ACCENT-colored "new" row produced ACCENT text on an
    # ACCENT background, unreadable. SECOND_HOVER never collides with any
    # tag's foreground (ACCENT/SECOND_FG/DIM), so it stays legible no
    # matter which tag the selected row has.
    style.map(
        "Treeview",
        background=[("selected", p["SECOND_HOVER"])],
        foreground=[("selected", p["SECOND_FG"])],
    )
    style.configure(
        "Treeview.Heading",
        background=p["SECOND_BG"], foreground=p["SECOND_FG"], font=FONT_BOLD,
        borderwidth=1, relief="flat",
    )
    style.map("Treeview.Heading", background=[("active", p["SECOND_HOVER"])])

    # Without this, ttk's "clam" theme default (a fixed neutral gray)
    # ignores the palette entirely, so the scrollbar visibly stayed the
    # same gray across both themes — confirmed directly, one of a few
    # spots that weren't actually "shifting" on toggle.
    style.configure(
        "Vertical.TScrollbar",
        background=p["SECOND_BG"], troughcolor=p["PANEL"], bordercolor=p["PANEL_BORDER"],
        arrowcolor=p["FG"],
    )
    style.map("Vertical.TScrollbar", background=[("active", p["SECOND_HOVER"])])


class ThemedWidgetRegistry:
    """Tracks plain tk widgets (Frame/Label/Text/etc.) that were given
    explicit bg=/fg= colors at construction time, so a theme switch can
    walk this list and reconfigure them — ttk widgets don't need this
    since re-running build_ttk_style() alone restyles every one of them.
    """

    def __init__(self):
        self._entries = []  # (widget, {option_name: palette_key})

    def add(self, widget, **option_to_key):
        self._entries.append((widget, option_to_key))
        return widget

    def refresh(self, palette):
        for widget, mapping in self._entries:
            try:
                widget.configure(**{opt: palette[key] for opt, key in mapping.items()})
            except Exception:
                # Widget may have been destroyed (e.g. a closed dialog) —
                # skip it rather than let one stale entry break the rest.
                pass
