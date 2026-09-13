"""Phase 1 disposable-but-real skeleton: one ugly screen, one button per
unverified assumption in the Android port plan (see
C:\\Users\\VivoBook\\.claude\\plans\\calm-waddling-sun.md). Not meant to
look good — meant to prove, on a real device/emulator, that each risky
piece actually works before any real UI gets built on top of it. Every
button here calls straight into the SAME shared engine
(text_analyzer.py) the Windows desktop app uses, unchanged — nothing
here reimplements anything, the same "thin wrapper calling into shared
logic" discipline the rest of this app already follows.
"""
import os
import sys

# text_analyzer.py lives one directory up (the project root) -- add it
# to sys.path before anything here imports it, same trick
# vocabmaster/main.py already uses for the desktop build.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView


class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.picked_text = None

        self.status = Label(
            text="Ready.", size_hint_y=None, halign="left", valign="top",
            text_size=(None, None),
        )
        self.status.bind(texture_size=self._update_status_height)
        scroll = ScrollView(size_hint_y=0.6)
        scroll.add_widget(self.status)
        self.add_widget(scroll)

        for label, callback in (
            ("1. Check text_analyzer import", self.check_import),
            ("2. Check bundled dictionary access", self.check_dictionary),
            ("3. Pick a book (PDF/EPUB)", self.pick_book),
            ("4. Build glossary from picked book", self.build_glossary),
            ("5. Speak a word out loud", self.speak_word),
        ):
            btn = Button(text=label, size_hint_y=None, height=70)
            btn.bind(on_release=callback)
            self.add_widget(btn)

    def _update_status_height(self, instance, value):
        instance.height = value[1]
        instance.text_size = (instance.width, None)

    def log(self, msg):
        self.status.text += "\n" + msg
        print(msg)

    def check_import(self, _instance):
        try:
            import text_analyzer
            self.log(f"OK: text_analyzer imported ({text_analyzer.__file__})")
        except Exception as exc:
            self.log(f"FAIL (1): {exc!r}")

    def check_dictionary(self, _instance):
        try:
            import sqlite3

            import text_analyzer
            path = text_analyzer._offline_dict_path("de")
            if not path:
                self.log("FAIL (2): dict_de.sqlite3 not found via _resource_dir()")
                return
            conn = sqlite3.connect(path)
            count = conn.execute("SELECT COUNT(*) FROM dict").fetchone()[0]
            conn.close()
            self.log(f"OK: dict_de.sqlite3 found at {path}, {count} rows")
        except Exception as exc:
            self.log(f"FAIL (2): {exc!r}")

    def pick_book(self, _instance):
        try:
            from plyer import filechooser
            filechooser.open_file(on_selection=self._on_file_picked, filters=["*.pdf", "*.epub"])
        except Exception as exc:
            self.log(f"FAIL (3): {exc!r}")

    def _on_file_picked(self, selection):
        if not selection:
            self.log("No file picked.")
            return
        try:
            import text_analyzer

            from mobile_app import android_support
            picked = selection[0]
            real_path = android_support.resolve_to_tempfile(picked)
            text = text_analyzer.load_book_text(real_path)
            self.picked_text = text
            self.log(f"OK (3): extracted {len(text)} characters from {picked}")
        except Exception as exc:
            self.log(f"FAIL (3): {exc!r}")

    def build_glossary(self, _instance):
        if not self.picked_text:
            self.log("Pick a book first (step 3).")
            return
        try:
            import text_analyzer
            freqs = text_analyzer.word_frequencies(self.picked_text)
            words = sorted(freqs.keys())[:200]
            text_analyzer.ensure_wordnet()
            defined, undefined, _ = text_analyzer.build_glossary(words)
            sample = ", ".join(w for w, _d in defined[:10])
            self.log(f"OK (4): {len(defined)} defined, {len(undefined)} undefined. Sample: {sample}")
        except Exception as exc:
            self.log(f"FAIL (4): {exc!r}")

    def speak_word(self, _instance):
        try:
            import text_analyzer
            text_analyzer.speak_word("hello", "en")
            self.log("OK (5): speak_word('hello', 'en') completed without raising")
        except Exception as exc:
            self.log(f"FAIL (5): {exc!r}")


class TextAnalyzerToolchainTestApp(App):
    def build(self):
        return RootWidget()


if __name__ == "__main__":
    TextAnalyzerToolchainTestApp().run()
