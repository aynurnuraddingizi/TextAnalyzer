"""Android build entrypoint shim. python-for-android expects a main.py
at the project root (buildozer.spec's source.dir) -- the real app lives
in mobile_app/app.py; this file only exists to satisfy that path
convention, the same "thin shim reaching one directory over" trick
vocabmaster/main.py already uses for reaching text_analyzer.py. This
file is Android-build-only and never runs as part of the Windows
desktop workflow (vocabmaster/main.py is the desktop entrypoint).
"""
from mobile_app.app import TextAnalyzerToolchainTestApp

if __name__ == "__main__":
    TextAnalyzerToolchainTestApp().run()
