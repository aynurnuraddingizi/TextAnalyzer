"""Every bit of `pyjnius`-touching code for the Android build lives only
in this file — every function here is internally guarded (imports
`jnius`/`android` lazily, inside the function body, never at module
level) so this module stays safely importable on desktop too, even
though nothing here is ever actually called there.

Two jobs:
1. `resolve_to_tempfile(picked_path)` — Android's Storage Access
   Framework hands back a `content://...` URI, not a real filesystem
   path, when the user picks a file. `pypdf`/`ebooklib` (in
   text_analyzer.py's `load_pdf_text()`/`load_epub_text()`, both
   UNCHANGED for this) only understand real paths, so this copies the
   picked file's bytes into a real temp file in the app's own private
   storage first. If `picked_path` is already a real path (some
   plyer/Android version combinations hand one back directly instead
   of a content:// URI), it's returned unchanged — no copy needed.
2. Nothing else yet — this file grows in later phases (e.g. the
   ACTION_SEND share-sheet export hookup in Phase 4) but Phase 1 only
   needs the file-picker bridge.
"""
import os
import tempfile


def resolve_to_tempfile(picked_path):
    if not picked_path.startswith("content://"):
        return picked_path

    from jnius import autoclass

    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    Uri = autoclass("android.net.Uri")
    activity = PythonActivity.mActivity
    resolver = activity.getContentResolver()
    uri = Uri.parse(picked_path)
    input_stream = resolver.openInputStream(uri)

    # DISPLAY_NAME (for a sensible extension) isn't essential here since
    # load_book_text() branches on the ORIGINAL picked_path's extension
    # (see the caller in app.py), not this temp file's name — but a
    # readable suffix makes the temp directory easier to debug by hand.
    suffix = os.path.splitext(picked_path)[1] or ".bin"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)

    # pyjnius auto-converts a Python bytearray to/from a Java byte[]
    # argument, so a plain bytearray works directly as read()'s buffer
    # — no manual Java array construction needed.
    buf = bytearray(65536)
    with os.fdopen(fd, "wb") as f:
        while True:
            n = input_stream.read(buf)
            if n == -1:
                break
            f.write(bytes(buf[:n]))
    input_stream.close()
    return temp_path
