"""The "analyze this book" pipeline, extracted out of app.py's own
MainWindow.run_pipeline() so it has no Tkinter dependency and no
dependency on MainWindow's own attributes — just plain arguments in,
a plain tuple out (or an exception, left for the caller to catch).

Both vocabmaster/app.py (Tkinter, via a thin wrapper that forwards
run_pipeline_steps()'s callbacks onto its own threading.Thread +
queue.Queue) and the Android app (mobile_app/, via its own thin
wrapper forwarding onto a Kivy Clock-polled queue) call this same
function, so a future change to the pipeline itself — a new analysis
step, a reordering, a bug fix — only has to happen once and both apps
pick it up, rather than two hand-duplicated copies silently drifting.

`on_status`/`on_progress_setup`/`on_progress` are plain callbacks, not
a queue — the caller decides how status text/progress-bar updates
actually reach its own UI thread (app.py forwards each one straight
onto self.work_queue, exactly reproducing this function's pre-
extraction behavior).
"""
import time

from grammar_engine import analyze_grammar, ensure_pos_tagger
from text_analyzer import (
    OFFLINE_DICT_LANGS,
    WORDNET_LANG_CODES,
    build_cefr_map,
    build_example_sentences,
    build_glossary,
    build_pos_map,
    detect_language,
    ensure_phrase_data,
    ensure_wordnet,
    extract_phrases,
    readability_stats,
    split_sentences,
    word_frequencies,
)


def run_pipeline_steps(text, lang_choice, on_status, on_progress_setup, on_progress, should_continue):
    """Runs the full analysis pipeline on `text` and returns
    (defined, undefined, lang, freqs, examples, sentences, pos_map,
    cefr_map, grammar_results, readability, phrases) — exactly the
    payload app.py's poll_queue() unpacks from its "glossary_done"
    message. Raises on failure; the caller is responsible for catching,
    logging, and reporting that to its own UI, same as before this was
    extracted.
    """
    on_status("Extracting unique words...")
    freqs = word_frequencies(text)
    words = sorted(freqs.keys())
    lang = detect_language(text) if lang_choice == "auto" else lang_choice
    wordnet_lang = WORDNET_LANG_CODES.get(lang)
    offline_dict_lang = lang if lang in OFFLINE_DICT_LANGS else None
    # Split once, reused for both this run's example sentences and any
    # later Find in Book search — splitting is the expensive part, not
    # the search itself.
    sentences = split_sentences(text)
    examples = build_example_sentences(text, words, sentences=sentences)
    on_status(f"Found {len(words)} unique words. Checking dictionary data...")
    ensure_wordnet()
    on_progress_setup(len(words))

    last_status_at = [0.0]

    def on_progress_tick(i, total):
        on_progress(i, total)
        now = time.monotonic()
        if now - last_status_at[0] > 0.5:
            last_status_at[0] = now
            on_status(f"Looking up definitions... {i}/{total} words")

    defined, undefined, _ = build_glossary(
        words, on_progress=on_progress_tick, should_continue=should_continue,
        wordnet_lang=wordnet_lang, offline_dict_lang=offline_dict_lang,
    )
    # WordNet-only (English/Spanish/French/Arabic-via-WordNet) — German/
    # Turkish/Russian/most-Arabic simply get {} here since their offline
    # dictionaries carry no part-of-speech data at all (see
    # build_pos_map()'s docstring).
    pos_map = build_pos_map([w for w, _d in defined], wordnet_lang=wordnet_lang)
    # English/Spanish only (see build_cefr_map()'s docstring) — every
    # other language's `defined` list simply yields {} here.
    cefr_map = build_cefr_map([w for w, _d in defined], lang=lang, pos_map=pos_map)
    # Grammar analysis is the last step of this same run rather than a
    # separate button/thread: tagging every sentence with nltk's
    # lightweight perceptron tagger is expected to be in the same
    # low-single-digit-to-tens-of-seconds range this step's dictionary-
    # lookup loop above already runs in, and a second on-demand action
    # would split "analyze this book" into two separate waits/cancel-
    # paths for little benefit. English/Spanish only (see
    # analyze_grammar()'s docstring) — every other language gets {} for
    # free. Only English needs nltk's tagger downloaded first; Spanish
    # detection works from spelling/suffixes alone, no tagger involved.
    if lang in ("en", "es"):
        on_status("Analyzing grammar structures...")
    if lang == "en":
        ensure_pos_tagger()
    grammar_results = analyze_grammar(sentences, lang=lang, should_continue=should_continue)
    # English/Spanish only (see readability_stats()'s docstring) — every
    # other language gets None, same "quietly absent" convention as
    # pos_map/cefr_map/grammar_results.
    readability = readability_stats(text, lang=lang)
    # Language-general (see extract_phrases()'s docstring) — unlike
    # grammar/CEFR/pronunciation, this isn't English-only, so every book
    # gets phrase extraction, not just English/Spanish ones (though only
    # those two get the "recognized" real-dictionary tier — every
    # language still gets "recurring").
    ensure_phrase_data(lang)
    phrases = extract_phrases(sentences, lang=lang)
    return defined, undefined, lang, freqs, examples, sentences, pos_map, cefr_map, grammar_results, readability, phrases
