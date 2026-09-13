"""Entry point.

HOW TO RUN:
    python vocabmaster/main.py

REQUIRES: same as text_analyzer.py — see that file's docstring.
"""
import ctypes
import os
import sys

# text_analyzer.py lives one directory up (the project root), not inside
# this package — add it to sys.path before anything here imports it.
# PyInstaller resolves this differently (its own frozen import system,
# driven by pathex in TextAnalyzer.spec), so this only matters when
# running from source.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    # Without this, Windows DPI-virtualizes a non-DPI-aware app on any
    # scaled display (125%/150%/etc. — common on most laptops) by
    # stretching the whole rendered window as a bitmap: every font and
    # button turns visibly blurry, and the window occupies more physical
    # screen space than its own layout math accounts for. Confirmed
    # directly: on a 125%-scaled 1920x1080 display, an unaware window
    # sized to fit 1536x864 logical pixels rendered stretched to ~1920
    # physical pixels wide, overflowing the visible desktop even
    # maximized, despite every widget's own position/width being
    # correctly within bounds. Must be set before any Tk window is
    # created. PROCESS_PER_MONITOR_DPI_AWARE = 2.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

from app import MainWindow


def run():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    run()
