"""Glossary export writers: TXT/DOCX/PDF (moved from the original
vocabulary_app.py, unchanged logic) plus new CSV/JSON/Markdown writers.

Every writer takes the same plain-data arguments (no dependency on any
GUI class) so they're independently testable and reusable.
"""
import csv
import json
from xml.sax.saxutils import escape as xml_escape

from docx import Document
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from grammar_analyzer import CATEGORY_ORDER, CATEGORY_ORDER_ES


def coverage_line(freqs, top_n=200):
    """A concrete, motivating number grounded in how the book actually
    reads: thanks to Zipf's law, a fairly small set of the most frequent
    words accounts for a large share of a text's actual word
    occurrences, so this tells the reader roughly how much of the book
    they'd recognize by learning just the top N words.
    """
    total = sum(freqs.values())
    if not freqs or not total:
        return None
    n = min(top_n, len(freqs))
    top_sum = sum(sorted(freqs.values(), reverse=True)[:n])
    pct = round(top_sum / total * 100)
    return f"Learning the top {n} most frequent words here covers ~{pct}% of the book's running text."


def sorted_entries(entries, freqs, sort_by_freq):
    if sort_by_freq:
        return sorted(entries, key=lambda e: -freqs.get(e[0], 0))
    return entries


def status_of(word, known_words, learning_words):
    if word in known_words:
        return "known"
    if word in learning_words:
        return "learning"
    return "new"


def combined_tag(word, pos_map, cefr_map=None):
    """"noun, A1" (etc.) — part of speech and CEFR level joined for
    display, skipping whichever one (or both) `word` has no entry for.
    Every word from a purely offline-dictionary-backed language (German/
    Turkish/Russian/most Arabic) has no part of speech at all (see
    build_pos_map()'s docstring in text_analyzer.py), and CEFR level is
    English-only (see build_cefr_map()'s docstring) — both simply drop
    out of the joined string rather than showing as blank/None.
    """
    pos_map = pos_map or {}
    cefr_map = cefr_map or {}
    parts = [p for p in (pos_map.get(word), cefr_map.get(word)) if p]
    return ", ".join(parts)


def tag_prefix(word, pos_map, cefr_map=None):
    """"[noun, A1] " (etc.) to prepend to a definition in the plain-
    text-style exports (TXT/DOCX/PDF/Markdown/Anki) — or "" when
    combined_tag() has nothing for `word`. CSV/JSON get real separate
    fields instead of this text prefix, since they're structured data,
    not prose.
    """
    tag = combined_tag(word, pos_map, cefr_map)
    return f"[{tag}] " if tag else ""


def export_txt(path, filepath, entries, undefined_words, freqs, examples, sort_by_freq, pos_map=None, cefr_map=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("VOCABULARY GLOSSARY\n")
        f.write(f"Source file: {filepath}\n")
        f.write(f"Words with a definition: {len(entries)}\n")
        f.write(f"Words with no dictionary match: {len(undefined_words)}\n")
        cov = coverage_line(freqs)
        if cov:
            f.write(f"{cov}\n")
        f.write("\nDEFINITIONS (most frequent first)\n" if sort_by_freq else "\nDEFINITIONS\n")
        for word, definition in sorted_entries(entries, freqs, sort_by_freq):
            count = freqs.get(word)
            suffix = f"  ({count}x)" if count else ""
            f.write(f"  {word}{suffix}: {tag_prefix(word, pos_map, cefr_map)}{definition}\n")
            example = examples.get(word)
            if example:
                f.write(f"      “{example}”\n")
        if undefined_words:
            f.write("\nWORDS WITH NO DICTIONARY MATCH\n")
            for word in undefined_words:
                f.write(f"  {word}\n")


def export_docx(path, filepath, entries, undefined_words, freqs, examples, sort_by_freq, pos_map=None, cefr_map=None):
    doc = Document()
    doc.add_heading("Vocabulary Glossary", level=1)
    doc.add_paragraph(f"Source file: {filepath}")
    doc.add_paragraph(f"Words with a definition: {len(entries)}")
    doc.add_paragraph(f"Words with no dictionary match: {len(undefined_words)}")
    cov = coverage_line(freqs)
    if cov:
        doc.add_paragraph(cov)

    doc.add_heading("Definitions" + (" (most frequent first)" if sort_by_freq else ""), level=2)
    for word, definition in sorted_entries(entries, freqs, sort_by_freq):
        count = freqs.get(word)
        p = doc.add_paragraph()
        p.add_run(word + (f"  ({count}x)" if count else "")).bold = True
        p.add_run(f": {tag_prefix(word, pos_map, cefr_map)}{definition}")
        example = examples.get(word)
        if example:
            ex = doc.add_paragraph()
            ex.add_run(f"“{example}”").italic = True

    if undefined_words:
        doc.add_heading("Words with no dictionary match", level=2)
        doc.add_paragraph(", ".join(undefined_words))

    doc.save(path)


def export_pdf(path, filepath, entries, undefined_words, freqs, examples, sort_by_freq, pos_map=None, cefr_map=None):
    doc = SimpleDocTemplate(path, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Vocabulary Glossary", styles["Title"]),
        Paragraph(f"Source file: {xml_escape(filepath)}", styles["Normal"]),
        Paragraph(f"Words with a definition: {len(entries)}", styles["Normal"]),
        Paragraph(f"Words with no dictionary match: {len(undefined_words)}", styles["Normal"]),
    ]
    cov = coverage_line(freqs)
    if cov:
        story.append(Paragraph(xml_escape(cov), styles["Normal"]))
    story.append(Spacer(1, 12))
    for word, definition in sorted_entries(entries, freqs, sort_by_freq):
        count = freqs.get(word)
        label = word + (f"  ({count}x)" if count else "")
        text = f"{tag_prefix(word, pos_map, cefr_map)}{definition}"
        story.append(Paragraph(f"<b>{xml_escape(label)}</b>: {xml_escape(text)}", styles["Normal"]))
        example = examples.get(word)
        if example:
            story.append(Paragraph(f"<i>“{xml_escape(example)}”</i>", styles["Normal"]))
        story.append(Spacer(1, 4))

    if undefined_words:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Words with no dictionary match", styles["Heading2"]))
        story.append(Paragraph(xml_escape(", ".join(undefined_words)), styles["Normal"]))

    doc.build(story)


def export_csv(path, filepath, entries, undefined_words, freqs, examples, sort_by_freq,
                known_words=(), learning_words=(), pos_map=None, cefr_map=None):
    pos_map = pos_map or {}
    cefr_map = cefr_map or {}
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count", "part_of_speech", "cefr_level", "definition", "example", "status"])
        for word, definition in sorted_entries(entries, freqs, sort_by_freq):
            writer.writerow([
                word, freqs.get(word, ""), pos_map.get(word, ""), cefr_map.get(word, ""), definition,
                examples.get(word, ""), status_of(word, known_words, learning_words),
            ])
        for word in undefined_words:
            writer.writerow([word, freqs.get(word, ""), "", "", "", examples.get(word, ""), "undefined"])


def export_json(path, filepath, entries, undefined_words, freqs, examples, sort_by_freq, lang,
                 known_words=(), learning_words=(), pos_map=None, cefr_map=None):
    pos_map = pos_map or {}
    cefr_map = cefr_map or {}
    data = {
        "source_file": filepath,
        "language": lang,
        "coverage_note": coverage_line(freqs),
        "words": [
            {
                "word": word,
                "count": freqs.get(word),
                "part_of_speech": pos_map.get(word),
                "cefr_level": cefr_map.get(word),
                "definition": definition,
                "example": examples.get(word),
                "status": status_of(word, known_words, learning_words),
            }
            for word, definition in sorted_entries(entries, freqs, sort_by_freq)
        ],
        "undefined": list(undefined_words),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_anki(path, entries, freqs, examples, sort_by_freq, known_words, learning_words, deck_name,
                 pos_map=None, cefr_map=None):
    """Write a tab-separated text file Anki's own text importer (File >
    Import, or just double-clicking the file with Anki installed) reads
    directly — no `genanki` dependency, which isn't installed and would
    add untested weight to the already-verified PyInstaller build for a
    feature that a plain text format handles just as well.

    The `#`-prefixed header lines are directives Anki 2.1's importer
    recognizes and uses to pre-fill the import dialog (deck, note type,
    separator, HTML rendering) so the file mostly imports itself instead
    of requiring manual column mapping.

    Already-known words are skipped: a flashcard for a word the reader
    has already marked known has no study value.

    A word's CEFR level (English only — see build_cefr_map()'s
    docstring), if any, is added to the note's own Tags field alongside
    its known/learning/new status, so the deck can be filtered/browsed
    by level directly inside Anki, not just by study status.
    """
    def esc(text):
        return xml_escape(text or "").replace("\t", " ").replace("\n", "<br>")

    safe_deck = (deck_name or "Vocabulary").replace("\t", " ").replace("\n", " ")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#separator:tab\n")
        f.write("#html:true\n")
        f.write("#notetype:Basic\n")
        f.write(f"#deck:{safe_deck}\n")
        f.write("#columns:Front\tBack\tTags\n")
        pos_map = pos_map or {}
        cefr_map = cefr_map or {}
        for word, definition in sorted_entries(entries, freqs, sort_by_freq):
            if word in known_words:
                continue
            tag_label = combined_tag(word, pos_map, cefr_map)
            back = f"<i>({tag_label})</i> {esc(definition)}" if tag_label else esc(definition)
            example = examples.get(word)
            if example:
                back += f"<br><br><i>“{esc(example)}”</i>"
            status_tag = status_of(word, known_words, learning_words)
            level = cefr_map.get(word)
            tags = f"{status_tag} {level}" if level else status_tag
            f.write(f"{esc(word)}\t{back}\t{tags}\n")


def export_markdown(path, filepath, entries, undefined_words, freqs, examples, sort_by_freq, pos_map=None, cefr_map=None):
    lines = ["# Vocabulary Glossary", "", f"**Source file:** {filepath}  ",
             f"**Words with a definition:** {len(entries)}  ",
             f"**Words with no dictionary match:** {len(undefined_words)}  "]
    cov = coverage_line(freqs)
    if cov:
        lines.append(f"**{cov}**  ")
    lines.append("")
    lines.append("## Definitions" + (" (most frequent first)" if sort_by_freq else ""))
    lines.append("")
    for word, definition in sorted_entries(entries, freqs, sort_by_freq):
        count = freqs.get(word)
        suffix = f" ({count}x)" if count else ""
        tag_label = combined_tag(word, pos_map, cefr_map)
        tag_note = f" *({tag_label})*" if tag_label else ""
        lines.append(f"- **{word}**{suffix}{tag_note} — {definition}")
        example = examples.get(word)
        if example:
            lines.append(f"  > {example}")
    if undefined_words:
        lines.append("")
        lines.append("## Words with no dictionary match")
        lines.append("")
        lines.append(", ".join(undefined_words))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------- grammar export writers

def _phrase_sections(phrases):
    """[(section_heading, entry), ...] flattening extract_phrases()'s
    two-tier {"recognized": [...], "recurring": {length: {...}}} shape
    into one ordered sequence, in the exact same grouping/order the
    Phrases tree itself uses (phrase_analyzer.populate_phrase_tree()):
    Idioms, then Phrasal Verbs & Fixed Expressions (both from
    "recognized", split on its "idiomatic" flag), then each length's
    phrases from "recurring", shortest first. A section with nothing in
    it is simply skipped, same "don't show what wasn't there"
    convention used throughout this app.
    """
    recognized = phrases.get("recognized") or []
    idioms = [e for e in recognized if e["idiomatic"]]
    others = [e for e in recognized if not e["idiomatic"]]
    sections = []
    if idioms:
        sections.append(("Idioms", idioms))
    if others:
        sections.append(("Other Phrases", others))
    recurring = phrases.get("recurring") or {}
    for length in sorted(recurring.keys()):
        entries = recurring[length]["entries"]
        if entries:
            sections.append((f"{length}-word phrases", entries))
    ordered = []
    for heading, entries in sections:
        for entry in entries:
            ordered.append((heading, entry))
    return ordered


def _phrase_line_info(entry):
    """(count_note, body_text) for one phrase entry, handling both
    shapes _phrase_entry_by_iid()'s docstring (app.py) describes:
    "recognized" entries (real dictionary definition, no PMI) and
    "recurring" ones (optional WordNet "meaning", optional PMI).
    `count_note` is the "(3x)" / "(3x, PMI 5.12)" suffix every writer
    below puts next to the phrase itself; `body_text` is whichever real
    text (definition or meaning) is available, or "" if neither is.
    """
    if "idiomatic" in entry:
        return f"({entry['count']}x)", entry["definition"]
    pmi_note = f", PMI {entry['pmi']}" if entry.get("pmi") is not None else ""
    return f"({entry['count']}x{pmi_note})", entry.get("meaning") or ""


def export_phrases_csv(path, filepath, phrases):
    # Every matching sentence, not just one — same "list every real
    # occurrence" choice export_grammar_csv() makes, for the same reason:
    # a phrase's whole value here is seeing every real use in the book.
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "phrase", "pos", "idiomatic", "count", "pmi", "definition", "example"])
        for heading, entry in _phrase_sections(phrases):
            is_recognized = "idiomatic" in entry
            writer.writerow([
                heading, entry["phrase"], entry.get("pos", ""), is_recognized and entry["idiomatic"],
                entry["count"], "" if is_recognized else (entry.get("pmi") or ""),
                entry["definition"] if is_recognized else (entry.get("meaning") or ""), entry["example"],
            ])


def export_phrases_json(path, filepath, phrases, lang):
    entries_out = []
    for heading, entry in _phrase_sections(phrases):
        is_recognized = "idiomatic" in entry
        entries_out.append({
            "section": heading, "phrase": entry["phrase"], "pos": entry.get("pos"),
            "idiomatic": bool(is_recognized and entry["idiomatic"]), "count": entry["count"],
            "pmi": None if is_recognized else entry.get("pmi"),
            "definition": entry["definition"] if is_recognized else entry.get("meaning"),
            "example": entry["example"],
        })
    data = {"source_file": filepath, "language": lang, "phrases": entries_out}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_phrases_txt(path, filepath, phrases):
    with open(path, "w", encoding="utf-8") as f:
        f.write("PHRASE ANALYSIS\n")
        f.write(f"Source file: {filepath}\n")
        sections = _phrase_sections(phrases)
        f.write(f"Phrases found: {len(sections)}\n")
        current_heading = None
        for heading, entry in sections:
            if heading != current_heading:
                current_heading = heading
                f.write(f"\n{heading.upper()}\n")
            count_note, body = _phrase_line_info(entry)
            f.write(f"\n  {entry['phrase']}  {count_note}\n")
            if body:
                f.write(f"    {body}\n")
            f.write(f"    e.g. “{entry['example']}”\n")


def export_phrases_anki(path, phrases, deck_name):
    """One card per phrase: front is the phrase itself, back is its real
    dictionary definition ("recognized" entries) or WordNet "meaning"
    when extract_phrases() found one ("recurring" entries — see its own
    docstring) plus a real example sentence from the book, mirroring
    export_grammar_anki()'s "definition + a few illustrative sentences,
    not every occurrence" card shape. No known/learning skip logic here
    (unlike the vocabulary/grammar Anki exports) — Phrases has no know/
    learning tracking of its own (see
    phrase_analyzer.build_phrase_section()'s docstring for why), so
    every extracted phrase is included. The section heading becomes the
    note's Anki tag, so a deck can be filtered/browsed by Idioms vs.
    Phrasal Verbs vs. length the same way grammar structures can by
    category.
    """
    def esc(text):
        return xml_escape(text or "").replace("\t", " ").replace("\n", "<br>")

    safe_deck = (deck_name or "Phrases").replace("\t", " ").replace("\n", " ")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#separator:tab\n")
        f.write("#html:true\n")
        f.write("#notetype:Basic\n")
        f.write(f"#deck:{safe_deck}\n")
        f.write("#columns:Front\tBack\tTags\n")
        for heading, entry in _phrase_sections(phrases):
            _count_note, body = _phrase_line_info(entry)
            back = esc(body) if body else "<i>(no dictionary meaning found)</i>"
            back += f"<br><br><i>“{esc(entry['example'])}”</i>"
            tag = heading.lower().replace(" ", "_").replace("&", "and")
            f.write(f"{esc(entry['phrase'])}\t{back}\t{tag}\n")


def export_phrases_docx(path, filepath, phrases):
    doc = Document()
    doc.add_heading("Phrase Analysis", level=1)
    doc.add_paragraph(f"Source file: {filepath}")
    sections = _phrase_sections(phrases)
    doc.add_paragraph(f"Phrases found: {len(sections)}")

    current_heading = None
    for heading, entry in sections:
        if heading != current_heading:
            current_heading = heading
            doc.add_heading(heading, level=2)
        count_note, body = _phrase_line_info(entry)
        p = doc.add_paragraph()
        p.add_run(f"{entry['phrase']}  {count_note}").bold = True
        if body:
            doc.add_paragraph(body)
        ex_p = doc.add_paragraph()
        ex_p.add_run(f"“{entry['example']}”").italic = True

    doc.save(path)


def export_phrases_pdf(path, filepath, phrases):
    doc = SimpleDocTemplate(path, pagesize=LETTER)
    styles = getSampleStyleSheet()
    sections = _phrase_sections(phrases)
    story = [
        Paragraph("Phrase Analysis", styles["Title"]),
        Paragraph(f"Source file: {xml_escape(filepath)}", styles["Normal"]),
        Paragraph(f"Phrases found: {len(sections)}", styles["Normal"]),
        Spacer(1, 12),
    ]
    current_heading = None
    for heading, entry in sections:
        if heading != current_heading:
            current_heading = heading
            story.append(Paragraph(xml_escape(heading), styles["Heading2"]))
        count_note, body = _phrase_line_info(entry)
        label = f"{entry['phrase']}  {count_note}"
        story.append(Paragraph(f"<b>{xml_escape(label)}</b>", styles["Normal"]))
        if body:
            story.append(Paragraph(xml_escape(body), styles["Normal"]))
        story.append(Paragraph(f"<i>“{xml_escape(entry['example'])}”</i>", styles["Normal"]))
        story.append(Spacer(1, 8))
    doc.build(story)


def export_phrases_markdown(path, filepath, phrases):
    sections = _phrase_sections(phrases)
    lines = ["# Phrase Analysis", "", f"**Source file:** {filepath}  ", f"**Phrases found:** {len(sections)}  ", ""]
    current_heading = None
    for heading, entry in sections:
        if heading != current_heading:
            current_heading = heading
            lines.append(f"## {heading}")
            lines.append("")
        count_note, body = _phrase_line_info(entry)
        lines.append(f"### {entry['phrase']}  {count_note}")
        lines.append("")
        if body:
            lines.append(body)
            lines.append("")
        lines.append(f"*“{entry['example']}”*")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _sorted_grammar_rules(grammar_results):
    """[(rule_id, data), ...] ordered the same way the Grammar tree
    already groups them (CATEGORY_ORDER or, for a Spanish result set,
    CATEGORY_ORDER_ES — see below — then alphabetically by name within
    a category) — so an exported document's structure order matches
    what the reader already saw on screen, not an arbitrary
    dict-iteration order.

    Which order applies is detected from `grammar_results` itself
    rather than needing a `lang` argument threaded through every
    export_grammar_*() caller: English and Spanish category names are
    entirely disjoint sets, and one result set is always from a single
    book's single detected language (analyze_grammar() never mixes
    both), so checking whether any category present is one of
    CATEGORY_ORDER_ES's is a reliable, simpler stand-in for asking the
    caller which language this was.
    """
    categories_present = {data["category"] for data in grammar_results.values()}
    order = CATEGORY_ORDER_ES if categories_present & set(CATEGORY_ORDER_ES) else CATEGORY_ORDER
    category_rank = {c: i for i, c in enumerate(order)}
    return sorted(
        grammar_results.items(),
        key=lambda item: (category_rank.get(item[1]["category"], len(order)), item[1]["name"]),
    )


def export_grammar_csv(path, filepath, grammar_results, known_rules=(), learning_rules=()):
    # Unlike the vocabulary CSV's single "example" column, every
    # matching sentence is included here (joined into one cell) —
    # listing every occurrence, not just one example, is the whole
    # point of the Grammar feature the reader asked for.
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "structure", "status", "sentence_count", "definition", "rule", "sentences"])
        for rule_id, data in _sorted_grammar_rules(grammar_results):
            writer.writerow([
                data["category"], data["name"], status_of(rule_id, known_rules, learning_rules),
                data["total_count"], data["definition"], data["rule_explanation"],
                " | ".join(data["sentences"]),
            ])


def export_grammar_json(path, filepath, grammar_results, lang, known_rules=(), learning_rules=()):
    data = {
        "source_file": filepath,
        "language": lang,
        "structures": [
            {
                "id": rule_id,
                "category": rule_data["category"],
                "name": rule_data["name"],
                "definition": rule_data["definition"],
                "rule": rule_data["rule_explanation"],
                "sentence_count": rule_data["total_count"],
                "sentences": rule_data["sentences"],
                "status": status_of(rule_id, known_rules, learning_rules),
            }
            for rule_id, rule_data in _sorted_grammar_rules(grammar_results)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_grammar_txt(path, filepath, grammar_results, known_rules=(), learning_rules=()):
    with open(path, "w", encoding="utf-8") as f:
        f.write("GRAMMAR ANALYSIS\n")
        f.write(f"Source file: {filepath}\n")
        f.write(f"Structures found: {len(grammar_results)}\n\n")
        current_category = None
        for rule_id, data in _sorted_grammar_rules(grammar_results):
            if data["category"] != current_category:
                current_category = data["category"]
                f.write(f"\n{current_category.upper()}\n")
            status = status_of(rule_id, known_rules, learning_rules)
            f.write(f"\n  {data['name']}  ({data['total_count']}x)  [{status}]\n")
            f.write(f"    {data['definition']}\n")
            f.write(f"    {data['rule_explanation']}\n")
            for i, sentence in enumerate(data["sentences"], start=1):
                f.write(f"      {i}. {sentence}\n")


def export_grammar_docx(path, filepath, grammar_results, known_rules=(), learning_rules=()):
    doc = Document()
    doc.add_heading("Grammar Analysis", level=1)
    doc.add_paragraph(f"Source file: {filepath}")
    doc.add_paragraph(f"Structures found: {len(grammar_results)}")

    current_category = None
    for rule_id, data in _sorted_grammar_rules(grammar_results):
        if data["category"] != current_category:
            current_category = data["category"]
            doc.add_heading(current_category, level=2)
        status = status_of(rule_id, known_rules, learning_rules)
        p = doc.add_paragraph()
        p.add_run(f"{data['name']}  ({data['total_count']}x)  [{status}]").bold = True
        doc.add_paragraph(data["definition"])
        rule_p = doc.add_paragraph()
        rule_p.add_run(data["rule_explanation"]).italic = True
        for i, sentence in enumerate(data["sentences"], start=1):
            doc.add_paragraph(f"{i}. {sentence}")

    doc.save(path)


def export_grammar_pdf(path, filepath, grammar_results, known_rules=(), learning_rules=()):
    doc = SimpleDocTemplate(path, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Grammar Analysis", styles["Title"]),
        Paragraph(f"Source file: {xml_escape(filepath)}", styles["Normal"]),
        Paragraph(f"Structures found: {len(grammar_results)}", styles["Normal"]),
        Spacer(1, 12),
    ]
    current_category = None
    for rule_id, data in _sorted_grammar_rules(grammar_results):
        if data["category"] != current_category:
            current_category = data["category"]
            story.append(Paragraph(xml_escape(current_category), styles["Heading2"]))
        status = status_of(rule_id, known_rules, learning_rules)
        label = f"{data['name']}  ({data['total_count']}x)  [{status}]"
        story.append(Paragraph(f"<b>{xml_escape(label)}</b>", styles["Normal"]))
        story.append(Paragraph(xml_escape(data["definition"]), styles["Normal"]))
        story.append(Paragraph(f"<i>{xml_escape(data['rule_explanation'])}</i>", styles["Normal"]))
        for i, sentence in enumerate(data["sentences"], start=1):
            story.append(Paragraph(xml_escape(f"{i}. {sentence}"), styles["Normal"]))
        story.append(Spacer(1, 8))
    doc.build(story)


def export_grammar_markdown(path, filepath, grammar_results, known_rules=(), learning_rules=()):
    lines = ["# Grammar Analysis", "", f"**Source file:** {filepath}  ",
             f"**Structures found:** {len(grammar_results)}  ", ""]
    current_category = None
    for rule_id, data in _sorted_grammar_rules(grammar_results):
        if data["category"] != current_category:
            current_category = data["category"]
            lines.append(f"## {current_category}")
            lines.append("")
        status = status_of(rule_id, known_rules, learning_rules)
        lines.append(f"### {data['name']}  ({data['total_count']}x)  *[{status}]*")
        lines.append("")
        lines.append(data["definition"])
        lines.append("")
        lines.append(f"*{data['rule_explanation']}*")
        lines.append("")
        for i, sentence in enumerate(data["sentences"], start=1):
            lines.append(f"{i}. {sentence}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_grammar_anki(path, grammar_results, known_rules, learning_rules, deck_name):
    """One card per RULE, not per sentence — a rule with 300 matching
    sentences would otherwise produce 300 near-duplicate cards, which
    has no real study value (see export_anki()'s own "front=word" model
    for the vocabulary equivalent of this same one-card-per-concept
    choice). The back shows the definition, the rule, and up to 3
    illustrative sentences — enough to jog recall without dumping the
    entire matched-sentence list onto one flashcard.

    Already-known structures are skipped, mirroring export_anki()'s own
    "no study value in a flashcard for something already known" logic.
    """
    def esc(text):
        return xml_escape(text or "").replace("\t", " ").replace("\n", "<br>")

    safe_deck = (deck_name or "Grammar").replace("\t", " ").replace("\n", " ")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#separator:tab\n")
        f.write("#html:true\n")
        f.write("#notetype:Basic\n")
        f.write(f"#deck:{safe_deck}\n")
        f.write("#columns:Front\tBack\tTags\n")
        for rule_id, data in _sorted_grammar_rules(grammar_results):
            if rule_id in known_rules:
                continue
            back = f"{esc(data['definition'])}<br><br><i>{esc(data['rule_explanation'])}</i>"
            for sentence in data["sentences"][:3]:
                back += f"<br><br><i>“{esc(sentence)}”</i>"
            tag = status_of(rule_id, known_rules, learning_rules)
            f.write(f"{esc(data['name'])}\t{back}\t{tag}\n")
