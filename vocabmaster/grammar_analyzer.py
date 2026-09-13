"""Grammar Analyzer ("Section 2"): detects which grammar structures
(tenses, mood, passive voice, modals, relative clauses, conditionals,
comparatives, non-finite forms, questions, and more) actually occur in
a loaded book's sentences, and surfaces them as a browsable
category/rule list — each with a plain-language definition, the
underlying rule, and every matching sentence. See build_grammar_section()
for the UI half of this; app.py wires analyze_grammar()'s output into it.

English and Spanish only. English uses nltk's bundled Penn Treebank POS
tagger (English-trained only — same reason meaningful part-of-speech
tagging and CEFR level are also English-only elsewhere; see
build_pos_map()/build_cefr_map()'s docstrings in text_analyzer.py) to
match POS-tag sequences against the _RULES catalog below. Spanish has
no such tagger available (nltk bundles none, and adding one would mean
a new heavy dependency this app has otherwise avoided) — see the
"Spanish" section further down, after _RULES, for the different,
suffix/lexical-anchor-based approach used there instead, and its own
detailed reasoning. Every other language gets {} from analyze_grammar()
below, same "decline rather than guess" choice as CEFR level.

Detection is POS-tag-sequence + lexical-anchor heuristics, not a real
syntactic parser. Every rule below states its own simplifying
assumption in its "rule_explanation" text where one matters, rather
than pretending to a precision this approach can't deliver. Calibrated
directly against nltk 3.9.1's actual tagger output on real sentences
(not assumed from documentation) while writing these:
  - Contractions split unusually: "won't" -> "wo" + "n't" (tagged
    MD + RB), "can't" -> "ca" + "n't", "'ll"/'ve"/"'re"/"'m" attach
    to the previous word as their own token. "'s" is genuinely
    ambiguous between "is" (be) and "has" (have); "'d" between "would"
    (modal) and "had" (past-perfect have). Both are listed in every
    word-set they could plausibly belong to (see BE_FORMS/HAVE_FORMS/
    MODAL_WORDS below) — the ambiguity resolves itself per rule, since
    e.g. present_progressive only fires when what follows is VBG and
    present_perfect only when what follows is VBN; the same contraction
    can never satisfy both in the same position.
  - The tagger sometimes mistags a sentence-initial capitalized -ing
    word as NNP/NN instead of VBG (e.g. "Walking home, ..." -> NNP), or
    an irregular past participle as VB instead of VBN (e.g. after "'d"
    in "She'd already left"). Both are accepted tagger noise this
    module doesn't work around — a rule simply won't fire on a sentence
    the tagger itself mis-tagged.

Spanish detection (see the "Spanish" section below, after the English
_RULES catalog) takes a completely different approach for a good
reason: nltk has no bundled Spanish POS tagger at all (the perceptron
tagger above is English-only), so there is no tag sequence to match
against in the first place. Spanish verb morphology is regular enough
— genuinely, not as a euphemism — that a suffix/lexical-anchor approach
built directly on that morphology stands in for tagging: see that
section's own docstring for the sourcing and the specific precision
trade-offs made there.
"""
import nltk
import re

from text_analyzer import _WORD_PATTERN

BE_FORMS = {"am", "is", "are", "was", "were", "be", "being", "been", "'s", "'m", "'re"}
PRESENT_BE = {"am", "is", "are", "'s", "'m", "'re"}
PAST_BE = {"was", "were"}
HAVE_FORMS = {"have", "has", "had", "'ve", "'s", "'d"}
PRESENT_HAVE = {"have", "has", "'ve", "'s"}
PAST_HAVE = {"had", "'d"}
MODAL_WORDS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought", "wo", "ca", "'ll", "'d"}
DO_FORMS = {"do", "does", "did"}
# Words that can act as an auxiliary/modal and so are excluded from the
# "genuine main verb" checks (present_simple/past_simple) — the known,
# documented gap this creates (a lexical use of have/be/do, e.g. "she
# HAS a book", is never counted as present_simple) is accepted, not
# solved: telling that apart from auxiliary use needs more than a POS
# tag carries.
AUX_LIKE = BE_FORMS | HAVE_FORMS | MODAL_WORDS | DO_FORMS
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}

_MAX_SENTENCE_LEN = 350  # same threshold text_analyzer.py's build_example_sentences()/find_occurrences() use
_MAX_SENTENCES_PER_RULE = 300  # mirrors concordance.py's MAX_RESULTS


def ensure_pos_tagger():
    """Make sure nltk's sentence tokenizer + POS tagger data are
    downloaded before use — same try/real-usage/except-LookupError/
    download shape as text_analyzer.ensure_wordnet(), just for a
    different pair of nltk data packages.

    Confirmed directly against nltk 3.9.1 (this project's installed
    version): the resource ids are "punkt_tab" and
    "averaged_perceptron_tagger_eng" — NOT the older
    "punkt"/"averaged_perceptron_tagger" names most documentation and
    tutorials still reference, which are stale for this nltk version.
    """
    try:
        nltk.word_tokenize("test")
    except LookupError:
        nltk.download("punkt_tab")
    try:
        nltk.pos_tag(["test"])
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng")


# --------------------------------------------------------------- helpers

def _word_at(tokens, i):
    return tokens[i][0].lower() if 0 <= i < len(tokens) else None


def _tag_at(tokens, i):
    return tokens[i][1] if 0 <= i < len(tokens) else None


def _tag_within(tokens, start, target_tags, window=2):
    """True if some token in tokens[start+1 : start+1+window] carries a
    tag in `target_tags` — the "allow a word or two of adverbs/negation
    in between" tolerance every auxiliary+verb rule below needs (e.g.
    "has already read", "hasn't quite finished")."""
    for j in range(start + 1, min(start + 1 + window, len(tokens))):
        if tokens[j][1] in target_tags:
            return True
    return False


def _is_going_to_future_at(tokens, i):
    """True if tokens[i] is the "going" of a be-going-to future (the
    token right after it is "to"/TO, then a base-form verb) — used by
    present_progressive/gerund to yield this specific shape to
    future_going_to instead of double-counting it as continuous aspect
    or a gerund."""
    return (
        _word_at(tokens, i) == "going"
        and _tag_at(tokens, i + 1) == "TO"
        and _tag_at(tokens, i + 2) == "VB"
    )


def _if_index(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() == "if":
            return i
    return None


# ------------------------------------------------------ tenses & aspect

def _detect_present_simple(tokens):
    return any(t in ("VBZ", "VBP") and w.lower() not in AUX_LIKE for w, t in tokens)


def _detect_present_progressive(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in PRESENT_BE:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if tokens[j][1] == "VBG" and not _is_going_to_future_at(tokens, j):
                    return True
    return False


def _detect_past_simple(tokens):
    return any(t == "VBD" and w.lower() not in AUX_LIKE for w, t in tokens)


def _detect_past_progressive(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in PAST_BE and _tag_within(tokens, i, {"VBG"}):
            return True
    return False


def _detect_present_perfect(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in PRESENT_HAVE:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if tokens[j][1] == "VBN" and _word_at(tokens, j) != "been":
                    return True
    return False


def _detect_present_perfect_progressive(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in PRESENT_HAVE:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _word_at(tokens, j) == "been" and _tag_within(tokens, j, {"VBG"}):
                    return True
    return False


def _detect_past_perfect(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in PAST_HAVE:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if tokens[j][1] == "VBN" and _word_at(tokens, j) != "been":
                    return True
    return False


def _detect_past_perfect_progressive(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in PAST_HAVE:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _word_at(tokens, j) == "been" and _tag_within(tokens, j, {"VBG"}):
                    return True
    return False


def _detect_future_will(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in ("will", "'ll", "wo") and _tag_within(tokens, i, {"VB"}):
            return True
    return False


def _detect_future_going_to(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in BE_FORMS:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _is_going_to_future_at(tokens, j):
                    return True
    return False


# ------------------------------------------------------------------ voice

def _detect_passive_voice(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in BE_FORMS and _tag_within(tokens, i, {"VBN"}):
            return True
    return False


# ---------------------------------------------------------------- modals

def _detect_modal_ability(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in ("can", "could", "ca") and _tag_within(tokens, i, {"VB"}):
            return True
    return False


def _detect_modal_obligation(tokens):
    for i, (w, t) in enumerate(tokens):
        wl = w.lower()
        if wl == "must" and t == "MD" and _tag_within(tokens, i, {"VB"}):
            return True
        if wl in HAVE_FORMS:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _word_at(tokens, j) == "to" and _tag_at(tokens, j + 1) == "VB":
                    return True
    return False


def _detect_modal_possibility(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() in ("may", "might") and _tag_within(tokens, i, {"VB"}):
            return True
    return False


def _detect_modal_advice(tokens):
    for i, (w, t) in enumerate(tokens):
        wl = w.lower()
        if wl == "should" and t == "MD" and _tag_within(tokens, i, {"VB"}):
            return True
        if wl == "ought" and _word_at(tokens, i + 1) == "to" and _tag_at(tokens, i + 2) == "VB":
            return True
    return False


def _detect_modal_would(tokens):
    for i, (w, t) in enumerate(tokens):
        if w.lower() in ("would", "'d") and t == "MD" and _tag_within(tokens, i, {"VB"}):
            return True
    return False


# ------------------------------------------------------------- clauses

REL_PRONOUNS = {"who", "whom", "whose", "which"}


def _detect_relative_clause(tokens):
    for i, (w, _t) in enumerate(tokens):
        wl = w.lower()
        if (wl in REL_PRONOUNS or wl == "that") and i > 0 and tokens[i - 1][1] in NOUN_TAGS:
            return True
    return False


def _detect_noun_clause_that(tokens):
    for i, (w, _t) in enumerate(tokens):
        if w.lower() == "that" and i > 0 and tokens[i - 1][1] in ("VB", "VBP", "VBZ", "VBD"):
            return True
    return False


def _detect_conditional_type1(tokens):
    idx = _if_index(tokens)
    if idx is None:
        return False
    has_present = any(t in ("VBZ", "VBP") and w.lower() not in AUX_LIKE for w, t in tokens[idx + 1:])
    return has_present and _detect_future_will(tokens)


def _detect_conditional_type2(tokens):
    idx = _if_index(tokens)
    if idx is None:
        return False
    # A VBD in the if-clause not immediately followed by VBN is treated
    # as this conditional's "simple past" ingredient, INCLUDING "had"
    # used as a main verb ("if I had more time") — only "had"/'d + VBN
    # (past perfect) is reserved for type 3 below.
    has_simple_past = any(
        t == "VBD" and _tag_at(tokens, j + 1) != "VBN"
        for j, (_w, t) in enumerate(tokens[idx + 1:], start=idx + 1)
    )
    return has_simple_past and _detect_modal_would(tokens)


def _detect_conditional_type3(tokens):
    idx = _if_index(tokens)
    if idx is None:
        return False
    has_past_perfect = any(
        w.lower() in PAST_HAVE and _tag_within(tokens, j, {"VBN"})
        for j, (w, _t) in enumerate(tokens[idx + 1:], start=idx + 1)
    )
    if not has_past_perfect:
        return False
    for i, (w, t) in enumerate(tokens):
        if w.lower() in ("would", "'d") and t == "MD":
            if _word_at(tokens, i + 1) == "have" and _tag_at(tokens, i + 2) == "VBN":
                return True
    return False


# --------------------------------------------------------------- comparison

def _detect_comparative(tokens):
    for i, (w, t) in enumerate(tokens):
        if t in ("JJR", "RBR"):
            return True
        if w.lower() in ("more", "less") and _tag_at(tokens, i + 1) in ("JJ", "RB"):
            return True
    return False


def _detect_superlative(tokens):
    for i, (w, t) in enumerate(tokens):
        if t in ("JJS", "RBS"):
            return True
        if w.lower() in ("most", "least") and _tag_at(tokens, i + 1) in ("JJ", "RB"):
            return True
    return False


# ---------------------------------------------------------- non-finite forms

def _detect_gerund(tokens):
    for i, (w, t) in enumerate(tokens):
        if t != "VBG":
            continue
        if w.lower() == "going" and _tag_at(tokens, i + 1) == "TO":
            continue  # claimed by future_going_to
        if _word_at(tokens, i - 1) in BE_FORMS:
            continue  # that's present/past progressive, not a gerund
        prev_tag = _tag_at(tokens, i - 1)
        if i == 0 or prev_tag in VERB_TAGS or prev_tag == "IN" or prev_tag in ("DT", "PRP$"):
            return True
    return False


def _detect_present_participle_phrase(tokens):
    for i, (_w, t) in enumerate(tokens):
        if t != "VBG":
            continue
        if _tag_at(tokens, i - 1) in NOUN_TAGS:
            return True  # reduced relative clause: "the man walking..."
        if i == 0:
            for j in range(1, min(5, len(tokens))):
                if tokens[j][0] == ",":
                    return True  # fronted participial phrase: "Smiling, she..."
    return False


def _detect_infinitive(tokens):
    for i, (w, t) in enumerate(tokens):
        if w.lower() == "to" and t == "TO" and _tag_at(tokens, i + 1) == "VB":
            return True
    return False


# -------------------------------------------------------------- questions

QUESTION_STARTERS = BE_FORMS | HAVE_FORMS | MODAL_WORDS | DO_FORMS
SUBJECT_TAGS = {"PRP", "NN", "NNS", "NNP", "NNPS", "DT"}


def _detect_yes_no_question(tokens):
    if not tokens or tokens[-1][0] != "?":
        return False
    return _word_at(tokens, 0) in QUESTION_STARTERS and _tag_at(tokens, 1) in SUBJECT_TAGS


def _detect_wh_question(tokens):
    if not tokens or tokens[-1][0] != "?":
        return False
    return _tag_at(tokens, 0) in ("WP", "WDT", "WRB", "WP$")


# ----------------------------------------------------------------- catalog

CATEGORY_ORDER = (
    "Tenses & Aspect", "Voice", "Modals", "Clauses & Conditionals",
    "Comparison", "Non-finite Forms", "Questions",
)

_RULES = [
    # ---- Tenses & Aspect ----
    {
        "id": "present_simple", "category": "Tenses & Aspect", "name": "Present Simple",
        "definition": "Used for habits, routines, and general truths — things that are regularly or always true.",
        "rule_explanation": "Subject + base verb (adds -s/-es for he/she/it): \"The sun rises in the east.\"",
        "detect": _detect_present_simple,
    },
    {
        "id": "present_progressive", "category": "Tenses & Aspect", "name": "Present Continuous",
        "definition": "Used for an action happening right now, or a temporary situation around the present.",
        "rule_explanation": "am/is/are + verb-ing: \"She is reading a book right now.\"",
        "detect": _detect_present_progressive,
    },
    {
        "id": "past_simple", "category": "Tenses & Aspect", "name": "Past Simple",
        "definition": "Used for a completed action or state in the past.",
        "rule_explanation": "Verb + -ed, or an irregular past form: \"She walked to school yesterday.\"",
        "detect": _detect_past_simple,
    },
    {
        "id": "past_progressive", "category": "Tenses & Aspect", "name": "Past Continuous",
        "definition": "Used for an action that was in progress at a specific past moment, often interrupted by another.",
        "rule_explanation": "was/were + verb-ing: \"She was reading when the phone rang.\"",
        "detect": _detect_past_progressive,
    },
    {
        "id": "present_perfect", "category": "Tenses & Aspect", "name": "Present Perfect",
        "definition": "Used for a past action with a connection to now — a result that still matters, or something not yet finished.",
        "rule_explanation": "have/has + past participle: \"She has already read that book.\"",
        "detect": _detect_present_perfect,
    },
    {
        "id": "present_perfect_progressive", "category": "Tenses & Aspect", "name": "Present Perfect Continuous",
        "definition": "Used for an action that started in the past and is still continuing, emphasizing its duration.",
        "rule_explanation": "have/has + been + verb-ing: \"She has been reading for an hour.\"",
        "detect": _detect_present_perfect_progressive,
    },
    {
        "id": "past_perfect", "category": "Tenses & Aspect", "name": "Past Perfect",
        "definition": "Used for an action that had already happened before another past action or time.",
        "rule_explanation": "had + past participle: \"She had already left when he arrived.\"",
        "detect": _detect_past_perfect,
    },
    {
        "id": "past_perfect_progressive", "category": "Tenses & Aspect", "name": "Past Perfect Continuous",
        "definition": "Used for an action that had been continuing up until a point in the past.",
        "rule_explanation": "had + been + verb-ing: \"She had been reading for an hour before he called.\"",
        "detect": _detect_past_perfect_progressive,
    },
    {
        "id": "future_will", "category": "Tenses & Aspect", "name": "Future with \"will\"",
        "definition": "Used for predictions, promises, and decisions made at the moment of speaking.",
        "rule_explanation": "will/'ll + base verb: \"She will call you tomorrow.\"",
        "detect": _detect_future_will,
    },
    {
        "id": "future_going_to", "category": "Tenses & Aspect", "name": "Future with \"going to\"",
        "definition": "Used for a plan or intention already decided before speaking, or a prediction based on present evidence.",
        "rule_explanation": "am/is/are + going to + base verb: \"She is going to visit her family.\"",
        "detect": _detect_future_going_to,
    },
    # ---- Voice ----
    {
        "id": "passive_voice", "category": "Voice", "name": "Passive Voice",
        "definition": "Used when the focus is on the person/thing the action happens to, rather than who does it.",
        "rule_explanation": (
            "A form of \"be\" + past participle: \"The cake was eaten before we arrived.\" Note: this "
            "heuristic also matches stative be+adjective phrases that merely look like a passive "
            "(\"I am interested in music\") — telling those apart needs more than a part-of-speech tag."
        ),
        "detect": _detect_passive_voice,
    },
    # ---- Modals ----
    {
        "id": "modal_ability", "category": "Modals", "name": "Modals of Ability",
        "definition": "\"can\"/\"could\" + base verb, used to say what someone is (or was) able to do. (Also sometimes used for permission, e.g. \"you can go\".)",
        "rule_explanation": "can/could + base verb: \"She can swim.\"",
        "detect": _detect_modal_ability,
    },
    {
        "id": "modal_obligation", "category": "Modals", "name": "Modals of Obligation",
        "definition": "\"must\", or \"have to\"/\"has to\"/\"had to\", used to say something is necessary or required.",
        "rule_explanation": "must + base verb, or have/has/had + to + base verb: \"You must finish this.\" / \"You have to finish this.\"",
        "detect": _detect_modal_obligation,
    },
    {
        "id": "modal_possibility", "category": "Modals", "name": "Modals of Possibility",
        "definition": "\"may\"/\"might\" + base verb, used to say something is possible but not certain. (Also sometimes used for permission, e.g. \"you may leave\".)",
        "rule_explanation": "may/might + base verb: \"It may rain later.\"",
        "detect": _detect_modal_possibility,
    },
    {
        "id": "modal_advice", "category": "Modals", "name": "Modals of Advice",
        "definition": "\"should\", or \"ought to\", used to give advice or say what's the right/expected thing to do.",
        "rule_explanation": "should + base verb, or ought + to + base verb: \"You should see a doctor.\"",
        "detect": _detect_modal_advice,
    },
    {
        "id": "modal_would", "category": "Modals", "name": "\"Would\"",
        "definition": "Used for hypothetical situations, past habits, and polite requests.",
        "rule_explanation": "would/'d + base verb: \"She would visit her grandmother every summer.\"",
        "detect": _detect_modal_would,
    },
    # ---- Clauses & Conditionals ----
    {
        "id": "relative_clause", "category": "Clauses & Conditionals", "name": "Relative Clauses",
        "definition": "A clause that gives more information about a noun, introduced by who/which/whose or (sometimes) \"that\".",
        "rule_explanation": (
            "Noun + who/which/whose/that + clause: \"The man who called is my brother.\" Note: \"that\" is also "
            "used to introduce a noun clause after a verb — see Noun Clauses."
        ),
        "detect": _detect_relative_clause,
    },
    {
        "id": "noun_clause_that", "category": "Clauses & Conditionals", "name": "Noun Clauses with \"that\"",
        "definition": "A clause acting as the object of a verb, introduced by \"that\" (often meaning \"the fact that...\").",
        "rule_explanation": (
            "Verb + that + clause: \"She said that she was tired.\" Note: this is the other common use of "
            "\"that\" — see Relative Clauses for the noun-modifying use."
        ),
        "detect": _detect_noun_clause_that,
    },
    {
        # Names kept short ("Conditional Type 1", not "...(Real/Future)")
        # since this app's Treeview columns clip overflow text with no
        # ellipsis — confirmed directly, long names were rendering as
        # visibly cut-off half-words. The Real/Future-style detail moves
        # into the definition text instead, where it's never clipped.
        "id": "conditional_type1", "category": "Clauses & Conditionals", "name": "Conditional Type 1",
        "definition": "The \"real/future\" conditional — used for a realistic condition and its likely future result.",
        "rule_explanation": "If + present simple, ... will + base verb: \"If it rains, I will stay home.\"",
        "detect": _detect_conditional_type1,
    },
    {
        "id": "conditional_type2", "category": "Clauses & Conditionals", "name": "Conditional Type 2",
        "definition": (
            "The \"unreal present\" conditional — used for an imaginary or unlikely present/future situation "
            "and its imagined result."
        ),
        "rule_explanation": "If + past simple, ... would + base verb: \"If I had more time, I would travel.\"",
        "detect": _detect_conditional_type2,
    },
    {
        "id": "conditional_type3", "category": "Clauses & Conditionals", "name": "Conditional Type 3",
        "definition": (
            "The \"unreal past\" conditional — used for an imaginary situation in the past and how things "
            "would have turned out differently."
        ),
        "rule_explanation": (
            "If + past perfect, ... would have + past participle: \"If she had studied, she would have passed.\" "
            "Note: this checks that these pieces appear together in the sentence, not real clause order or "
            "boundaries, so an unusual multi-clause sentence could occasionally match by coincidence."
        ),
        "detect": _detect_conditional_type3,
    },
    # ---- Comparison ----
    {
        "id": "comparative", "category": "Comparison", "name": "Comparatives",
        "definition": "Used to compare two things.",
        "rule_explanation": "adjective/adverb + -er, or more/less + adjective/adverb: \"This book is more interesting than that one.\"",
        "detect": _detect_comparative,
    },
    {
        "id": "superlative", "category": "Comparison", "name": "Superlatives",
        "definition": "Used to say something is the highest or lowest degree among three or more things.",
        "rule_explanation": "the + adjective/adverb + -est, or most/least + adjective/adverb: \"This is the most interesting book I have read.\"",
        "detect": _detect_superlative,
    },
    # ---- Non-finite Forms ----
    {
        "id": "gerund", "category": "Non-finite Forms", "name": "Gerunds",
        "definition": "The \"-ing\" form of a verb used as a noun — the subject or object of another verb.",
        "rule_explanation": (
            "verb/preposition/determiner + verb-ing (acting as a noun): \"I enjoy swimming every weekend.\" "
            "Note: English's -ing form is tagged the same way whether it's a true gerund (a noun) or a present "
            "participle (see Present Participle Phrases) — this rule uses the surrounding words as its best guess, "
            "not a real grammatical judgment."
        ),
        "detect": _detect_gerund,
    },
    {
        "id": "present_participle_phrase", "category": "Non-finite Forms", "name": "Present Participle Phrases",
        "definition": "An \"-ing\" phrase describing a noun, or added to the front of a sentence for a related action.",
        "rule_explanation": "noun + verb-ing... (\"the man walking down the street\"), or verb-ing, ... at the start of a sentence (\"Smiling, she opened the door.\")",
        "detect": _detect_present_participle_phrase,
    },
    {
        "id": "infinitive", "category": "Non-finite Forms", "name": "Infinitives",
        "definition": "\"to\" + base verb, used after another verb, adjective, or to express purpose.",
        "rule_explanation": (
            "to + base verb: \"She called to apologize.\" This covers both a purpose infinitive (\"to apologize\" "
            "= in order to apologize) and a complement infinitive (\"wants to leave\") — telling those apart needs "
            "knowledge of the specific verb involved, not just its part of speech."
        ),
        "detect": _detect_infinitive,
    },
    # ---- Questions ----
    {
        "id": "yes_no_question", "category": "Questions", "name": "Yes/No Questions",
        "definition": "A question that can be answered with \"yes\" or \"no\", formed by putting the auxiliary/modal verb first.",
        "rule_explanation": "Auxiliary/modal + subject + ...?: \"Is she coming?\"",
        "detect": _detect_yes_no_question,
    },
    {
        "id": "wh_question", "category": "Questions", "name": "Wh- Questions",
        "definition": "A question asking for specific information, starting with a question word.",
        "rule_explanation": "Who/what/where/when/why/how + ...?: \"Where are you going?\"",
        "detect": _detect_wh_question,
    },
]


# ===================================================================
# Spanish
# ===================================================================
#
# No POS tagger to lean on (see the module docstring above), so every
# rule below works directly from a sentence's own spelling: verb-ending
# suffixes, a curated table of the highest-frequency IRREGULAR verb
# forms (which don't follow those suffixes at all), and lexical anchors
# (question marks, "que", "si", reflexive/object pronouns, comparison
# words). All of it — endings, irregular paradigms, mood/tense
# semantics — is standard, textbook Spanish grammar, cross-checked
# against a regular-conjugation reference table during this feature's
# own development, not invented or guessed at.
#
# The single guiding precision rule, stated once here rather than
# repeated in every function below: SAFETY OVER COVERAGE. Several of
# Spanish's singular verb endings (especially "-o"/"-a"/"-e"/"-ía") are
# also ordinary noun/adjective endings ("casa", "parque", "alegría",
# and — the specific case that shaped this design — "panadería",
# "librería" and every other "-ería"/"-aría" shop-name noun, which
# would otherwise look exactly like a conditional-tense verb). Rather
# than accept those false positives, this module either (a) requires a
# LONGER, genuinely distinctive plural/vosotros ending no ordinary noun
# shares ("hablamos", "coméis", "vivían"), (b) requires an explicit
# subject pronoun immediately before an otherwise-ambiguous singular
# ending (a bare noun/adjective never directly follows "yo"/"tú"/"él"
# in grammatical Spanish, so "él habla" is safe even though "habla"
# alone isn't), or (c) falls back to an exact-match table of specific
# common irregular forms (safe because they're whole-word matches, not
# suffixes). The accepted cost is under-detection, not wrong detection
# — a regular verb's bare singular form used with no explicit subject
# pronoun nearby can go uncounted, same "decline rather than guess"
# choice this app makes everywhere else about real vs. invented data.

_ES_SUBJECT_PRONOUNS = {"yo", "tú", "tu", "él", "ella", "usted", "ud", "ello"}
_ES_PLURAL_SUBJECT_PRONOUNS = {"nosotros", "nosotras", "vosotros", "vosotras", "ellos", "ellas", "ustedes", "uds"}

# Minimum letters required before an ambiguous short ending is trusted
# at all (rejects tiny coincidental matches like "va" alone, which is
# already covered as a whole-word irregular form anyway).
_ES_MIN_STEM = 3


def _es_tokenize(sentence):
    return [t.lower() for t in re.findall(_WORD_PATTERN, sentence, re.UNICODE)]


def _es_ends_with_any(word, suffixes, min_len=0):
    return len(word) >= min_len and word.endswith(suffixes)


def _es_pronoun_before(tokens, i):
    return i > 0 and tokens[i - 1] in _ES_SUBJECT_PRONOUNS


# Words that typically introduce a NOUN phrase — immediately preceding
# one of these is the specific, narrow shape of the "-ería"/"-oría"
# noun trap this module's docstring describes ("la panadería", "una
# categoría"), used as a targeted REJECT-only check (see
# _es_ends_with_any_in_verb_position()) rather than the broader,
# stricter "require an explicit subject pronoun" used for endings with
# no known noun collision to guard against at all.
_ES_NOUN_PRECURSORS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "en", "a", "de", "del", "al"}


def _es_ends_with_any_in_verb_position(tokens, i, suffixes, min_len):
    """Like _es_ends_with_any(), plus: rejects a match immediately after
    a noun-precursor (see _ES_NOUN_PRECURSORS) — permissive everywhere
    else, INCLUDING a pro-dropped subject (sentence-initial, right after
    a comma, or after "que"/"si"), since Spanish drops subject pronouns
    far too often for requiring one to be workable. Used specifically
    for endings (the conditional's "-ría" family) with a real, narrow,
    identifiable noun-suffix collision to guard against — see
    _es_detect_condicional()'s own docstring for why this is a better
    trade-off there than either _es_pronoun_before()'s strictness or no
    guard at all.
    """
    word = tokens[i]
    if not _es_ends_with_any(word, suffixes, min_len):
        return False
    return not (i > 0 and tokens[i - 1] in _ES_NOUN_PRECURSORS)


# ---- irregular verb form tables (exact-match; safe by construction) ----
# Each maps a real, complete conjugated word form to nothing but its own
# existence — these are looked up as whole tokens, never as substrings,
# so there is no ambiguity to guard against the way there is for
# suffixes. Covers the highest-frequency irregular verbs (ser, estar,
# ir, haber, tener, hacer, poder, querer, decir, venir, dar, ver, saber,
# poner, salir) — the small set that, between them, accounts for a
# large share of all irregular-verb occurrences in ordinary text; a
# rarer irregular verb's forms that aren't listed here simply aren't
# specifically matched (they may still be caught by the regular-suffix
# rules for whichever tense they happen to superficially resemble, or
# not counted at all — under-detection, not a wrong answer).

_ES_PRESENTE_IRREGULAR = {
    "soy", "eres", "es", "somos", "sois", "son",
    "estoy", "estás", "está", "estamos", "estáis", "están",
    "voy", "vas", "va", "vamos", "vais", "van",
    "he", "has", "ha", "hemos", "habéis", "han", "hay",
    "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen",
    "hago", "haces", "hace", "hacemos", "hacéis", "hacen",
    "puedo", "puedes", "puede", "podemos", "podéis", "pueden",
    "quiero", "quieres", "quiere", "queremos", "queréis", "quieren",
    "digo", "dices", "dice", "decimos", "decís", "dicen",
    "vengo", "vienes", "viene", "venimos", "venís", "vienen",
    "doy", "das", "da", "damos", "dais", "dan",
    "veo", "ves", "ve", "vemos", "veis", "ven",
    "sé", "sabes", "sabe", "sabemos", "sabéis", "saben",
    "pongo", "pones", "pone", "ponemos", "ponéis", "ponen",
    "salgo", "sales", "sale", "salimos", "salís", "salen",
}
_ES_PRETERITO_IRREGULAR = {
    "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron",
    "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron",
    "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron",
    "hice", "hiciste", "hizo", "hicimos", "hicisteis", "hicieron",
    "pude", "pudiste", "pudo", "pudimos", "pudisteis", "pudieron",
    "quise", "quisiste", "quiso", "quisimos", "quisisteis", "quisieron",
    "dije", "dijiste", "dijo", "dijimos", "dijisteis", "dijeron",
    "vine", "viniste", "vino", "vinimos", "vinisteis", "vinieron",
    "di", "diste", "dio", "dimos", "disteis", "dieron",
    "vi", "viste", "vio", "vimos", "visteis", "vieron",
    "supe", "supiste", "supo", "supimos", "supisteis", "supieron",
    "puse", "pusiste", "puso", "pusimos", "pusisteis", "pusieron",
    "hube", "hubiste", "hubo", "hubimos", "hubisteis", "hubieron",
}
_ES_IMPERFECTO_IRREGULAR = {
    "era", "eras", "éramos", "erais", "eran",
    "iba", "ibas", "íbamos", "ibais", "iban",
    "veía", "veías", "veíamos", "veíais", "veían",
    # "querer"'s imperfecto is formation-regular (quer- + ía) but its
    # singular ends in "-ería" — confirmed directly this is genuinely
    # indistinguishable, by suffix alone, from ANY regular -er verb's
    # CONDICIONAL, which also ends in "-ería" (comer -> comería, temer
    # -> temería: condicional keeps the full infinitive, so "-er"+"ía"
    # always looks like this). Exact-matched here rather than solved in
    # general — querer is common enough that this specific collision is
    # worth handling by name, the same way "querría" is handled by name
    # on the condicional side (see _es_detect_condicional()) — a rarer
    # r-stem -er/-ir verb's own imperfecto singular isn't covered by
    # this and is accepted as a smaller residual gap.
    "quería", "querías",
}
_ES_SUBJUNTIVO_PRESENTE_IRREGULAR = {
    "sea", "seas", "seamos", "seáis", "sean",
    "esté", "estés", "estemos", "estéis", "estén",
    "vaya", "vayas", "vayamos", "vayáis", "vayan",
    "haya", "hayas", "hayamos", "hayáis", "hayan",
    "sepa", "sepas", "sepamos", "sepáis", "sepan",
    "dé", "des", "demos", "deis", "den",
    "tenga", "tengas", "tengamos", "tengáis", "tengan",
    "haga", "hagas", "hagamos", "hagáis", "hagan",
    "pueda", "puedas", "podamos", "podáis", "puedan",
    "quiera", "quieras", "queramos", "queráis", "quieran",
    "diga", "digas", "digamos", "digáis", "digan",
    "venga", "vengas", "vengamos", "vengáis", "vengan",
    "ponga", "pongas", "pongamos", "pongáis", "pongan",
    "salga", "salgas", "salgamos", "salgáis", "salgan",
}
# The 8 traditionally-irregular affirmative tú-commands, exceptions to
# the regular "drop the -s from tú's present tense" rule — e.g. "decir"
# would regularly give "dices" -> "dice", but the real command is "di".
_ES_IMPERATIVO_IRREGULAR = {"sé", "ve", "ten", "haz", "pon", "sal", "ven", "di"}

_ES_SER_FORMS = _ES_PRESENTE_IRREGULAR.intersection({
    "soy", "eres", "es", "somos", "sois", "son",
}) | {"era", "eras", "era", "éramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron",
      "sea", "seas", "sea", "seamos", "seáis", "sean", "sido", "siendo"}
_ES_ESTAR_FORMS = {
    "estoy", "estás", "está", "estamos", "estáis", "están",
    "estaba", "estabas", "estábamos", "estabais", "estaban",
    "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron",
    "esté", "estés", "estemos", "estéis", "estén", "estado", "estando",
}

_ES_REFLEXIVE_PRONOUNS = {"me", "te", "se", "nos", "os"}
_ES_OBJECT_PRONOUNS = {"lo", "la", "los", "las", "le", "les"}
_ES_QUESTION_WORDS = {
    "qué", "quién", "quiénes", "cómo", "cuándo", "dónde", "adónde",
    "cuál", "cuáles", "cuánto", "cuánta", "cuántos", "cuántas", "por qué",
}


def _es_verb_at(tokens, i, forms):
    return 0 <= i < len(tokens) and tokens[i] in forms


# Spanish's plural verb endings nest inside each other by spelling
# coincidence — "-íamos" (imperfecto) ends in "-amos" (presente/
# pretérito's own ending); "-aríamos"/"-eríamos"/"-iríamos"
# (condicional) end in "-íamos" in turn; "-remos"/"-réis" (futuro) end
# in "-emos"/"-éis" (presente). Checking each tense's ending in
# isolation — "does this word end in -amos?" — means a condicional
# form like "hablaríamos" trivially also passes the presente AND
# imperfecto checks, a real cascading false positive confirmed directly
# while testing this module (a single word firing four different tense
# tags at once). The fix is classifying each word's plural ending
# ONCE, longest (most specific) suffix first, rather than testing every
# tense's ending independently — ordered so a longer, unambiguous
# suffix is matched before a shorter one it happens to end with is ever
# checked. "-amos"/"-imos" are deliberately labeled for BOTH tenses:
# unlike every other collision here, presente and pretérito genuinely
# and correctly share identical spelling for -ar/-ir "nosotros" forms
# ("hablamos" truly can mean either "we speak" or "we spoke") — that
# one is real ambiguity in the language itself, not a detection flaw,
# so both tags are meant to fire together for it.
_ES_PLURAL_ENDING_CLASSES = (
    ("aríamos", ("condicional",)), ("eríamos", ("condicional",)), ("iríamos", ("condicional",)),
    ("aríais", ("condicional",)), ("eríais", ("condicional",)), ("iríais", ("condicional",)),
    ("arían", ("condicional",)), ("erían", ("condicional",)), ("irían", ("condicional",)),
    ("ábamos", ("imperfecto",)), ("íamos", ("imperfecto",)), ("íais", ("imperfecto",)), ("ían", ("imperfecto",)),
    ("abais", ("imperfecto",)), ("aban", ("imperfecto",)),
    ("remos", ("futuro",)), ("réis", ("futuro",)), ("rán", ("futuro",)),
    ("asteis", ("preterito",)), ("aron", ("preterito",)), ("isteis", ("preterito",)), ("ieron", ("preterito",)),
    ("amos", ("presente", "preterito")), ("imos", ("presente", "preterito")),
    ("áis", ("presente",)), ("emos", ("presente",)), ("éis", ("presente",)), ("ís", ("presente",)),
    # Deliberately NOT included: bare "-an"/"-en" (3rd person plural
    # presente for -ar/-er-ir) — confirmed directly this is unsafe,
    # unlike this list's other entries: "examen" (exam), "origen",
    # "joven", "orden", "crimen", "imagen", "margen", "resumen", and
    # "volumen" are all common, everyday Spanish NOUNS that happen to
    # end in "-en" too, with nothing else here to tell them apart from
    # a real "-en" verb form ("comen", "viven"). Presente detection for
    # this specific person/number combination is accepted as a real gap
    # rather than risk tagging "examen" as a verb.
)


def _es_classify_plural_ending(word):
    for suffix, tenses in _ES_PLURAL_ENDING_CLASSES:
        if len(word) >= _ES_MIN_STEM + len(suffix) and word.endswith(suffix):
            return tenses
    return ()


# ---------------------------------------------------- tenses & aspect


# The bare single-letter presente singular endings ("-o", "-a", "-e"...)
# are also each simply the LAST letter of several other tenses' own
# singular endings ("hablaría" ends in "-a" too; "hablaba" ends in
# "-a" too; "hablará"/"habló" end in "-a"-less vowels but "-e"/"-es"
# still collide with e.g. "comiste") — so before trusting one, this
# excludes any word that's more specifically explained by a DIFFERENT
# tense's own singular ending, the same "longer/more specific pattern
# wins" principle _es_classify_plural_ending() applies to plurals.
_ES_OTHER_TENSE_SINGULAR_ENDINGS = ("aba", "abas", "ía", "ías", "ría", "rías", "rá", "rás", "ré",
                                    "ó", "ió", "aste", "iste", "ara", "aras", "iera", "ieras",
                                    "ase", "ases", "iese", "ieses")


def _es_detect_presente(sentence, tokens):
    if any(t in _ES_PRESENTE_IRREGULAR for t in tokens):
        return True
    if any("presente" in _es_classify_plural_ending(t) for t in tokens):
        return True
    for i, t in enumerate(tokens):
        if not _es_pronoun_before(tokens, i):
            continue
        if _es_ends_with_any(t, _ES_OTHER_TENSE_SINGULAR_ENDINGS, 0):
            continue
        if _es_ends_with_any(t, ("o", "as", "a", "es", "e"), _ES_MIN_STEM):
            return True
    return False


def _es_detect_preterito(sentence, tokens):
    if any(t in _ES_PRETERITO_IRREGULAR for t in tokens):
        return True
    if any("preterito" in _es_classify_plural_ending(t) for t in tokens):
        return True
    # "-aste"/"-iste" (tú) and "-ó"/"-ió" (él/ella/usted) are trusted
    # unconditionally: unlike most of this section's singular endings,
    # no common Spanish noun or adjective ends in a stressed "-ó" or in
    # "-aste"/"-iste" — there's no equivalent of the "-ería" noun trap
    # (see _es_detect_condicional()) to guard against here.
    for t in tokens:
        if _es_ends_with_any(t, ("aste", "iste"), _ES_MIN_STEM + 3):
            return True
        if _es_ends_with_any(t, ("ó", "ió"), _ES_MIN_STEM + 1):
            return True
    return False


def _es_detect_imperfecto(sentence, tokens):
    if any(t in _ES_IMPERFECTO_IRREGULAR for t in tokens):
        return True
    if any("imperfecto" in _es_classify_plural_ending(t) for t in tokens):
        return True
    for t in tokens:
        if _es_ends_with_any(t, ("aba", "abas"), _ES_MIN_STEM + 3):
            return True
    for i, t in enumerate(tokens):
        if not _es_pronoun_before(tokens, i):
            continue
        # Exclude single-r "-ería"/"-iría"/"-aría": condicional's ending
        # literally CONTAINS imperfecto's own "-ía" as its tail
        # ("hablaría" ends in "ía" too), the same nested-suffix
        # coincidence _ES_PLURAL_ENDING_CLASSES exists to resolve for
        # plurals — condicional is the longer, more specific match and
        # wins UNLESS it's the double-r "-rría" shape ("corría"), which
        # is imperfecto far more often than it's "querer"'s irregular
        # condicional "querría" — see _es_detect_condicional()'s own
        # docstring on this specific ambiguity.
        if _es_ends_with_any(t, ("ería", "iría", "aría"), 0):
            continue
        if _es_ends_with_any(t, ("ía",), _ES_MIN_STEM + 2):
            return True
    return False


_ES_IRREGULAR_FUTURO_YO = {
    "tendré", "pondré", "saldré", "vendré", "valdré", "podré",
    "sabré", "cabré", "habré", "querré", "haré", "diré",
}


def _es_detect_futuro(sentence, tokens):
    # "-r" immediately before the ending is the universal future marker
    # regardless of conjugation class or irregular stem (every future
    # form is built on something ending in "r" — a regular infinitive
    # always does, and so, by design, does every irregular future stem:
    # "tendr-", "podr-", "dir-", "har-", etc.). "-rás"/"-rá" and the
    # plurals are trusted unconditionally — checked directly against
    # real Spanish vocabulary, no common noun/adjective ends that way.
    #
    # "-ré" (yo) needs real care, more than initially assumed: bare
    # "-ré" collides with regular preterito whenever an -ar verb's own
    # STEM happens to end in "r" ("comprar" -> "compré", preterito, NOT
    # "compraré" the real future — a different, longer word). Requiring
    # instead the fuller "-aré"/"-eré"/"-iré" (the infinitive's own
    # final vowel, always kept in a REGULAR future) looked like a fix,
    # but confirmed directly it isn't sufficient either: plenty of
    # common -ar verbs have a stem that itself happens to end in "ar"
    # or "er" ("preparar" -> stem "prepar" -> preterito "preparé" ends
    # in "aré"; "esperar" -> stem "esper" -> preterito "esperé" ends in
    # "eré") — real, common words, wrongly caught the same way. There
    # is no way to tell these apart from spelling alone without a verb
    # lexicon this app doesn't have, so regular singular futuro
    # requires an explicit "yo" nearby after all — a real recall cost,
    # accepted deliberately (same "safety over coverage" principle as
    # everywhere else in this section) rather than mislabel a
    # preterito verb as future. The dozen IRREGULAR future stems
    # (tendr-, podr-, etc.) don't have this problem — none of them is
    # also a valid preterito form of some OTHER verb — so those stay
    # unconditional, checked directly by their whole "yo"-form word.
    if any("futuro" in _es_classify_plural_ending(t) for t in tokens):
        return True
    if any(_es_ends_with_any(t, ("rás", "rá"), _ES_MIN_STEM + 2) for t in tokens):
        return True
    if any(t in _ES_IRREGULAR_FUTURO_YO for t in tokens):
        return True
    for i, t in enumerate(tokens):
        if t == "yo" and i + 1 < len(tokens) and _es_ends_with_any(tokens[i + 1], ("aré", "eré", "iré"), _ES_MIN_STEM + 3):
            return True
    return False


def _es_detect_condicional(sentence, tokens):
    # The plurals are safe outright via the classifier (no "-ríamos"/
    # "-ríais"/"-rían" noun forms exist to collide with). The singular
    # "-ría"/"-rías" has TWO real collisions to guard against:
    #   1. "panadería", "librería", "categoría", and every other
    #      "-ería"/"-oría" shop/abstract-noun word Spanish has — almost
    #      always appearing right after an article or preposition ("la
    #      panadería"); a genuine conditional verb essentially never
    #      does. See _es_ends_with_any_in_verb_position()'s own
    #      docstring for this trade-off's general shape.
    #   2. Confirmed directly while testing this module: imperfecto of
    #      any -er/-ir verb whose STEM itself ends in "r" — "correr"
    #      (to run) -> "corría" (was running), "querer" (to want) ->
    #      "quería" (wanted) — coincidentally ends in "-r"+"ía" the
    #      SAME way "querer"'s own irregular conditional "querría"
    #      does (querer's future/conditional stem is "querr-", with a
    #      double r), and unlike the noun trap above, there's no
    #      article/preposition signal to tell these apart — "corría"
    #      and "querría" are structurally identical shapes from
    #      different verbs. Since "querer" is the only one of this
    #      section's handful of known irregular stems this affects,
    #      it's handled by name (both here, positively, and — as
    #      "quería"/"querías", its imperfecto, which shares this
    #      collision from the OTHER side too — negatively, in
    #      _ES_IMPERFECTO_IRREGULAR); every other "-r"+"ía" word is
    #      treated as the (far more common) imperfecto reading instead.
    if any("condicional" in _es_classify_plural_ending(t) for t in tokens):
        return True
    for i, t in enumerate(tokens):
        t = tokens[i]
        if t in ("querría", "querrías"):
            return True
        if t in ("quería", "querías"):
            continue
        if _es_ends_with_any(t, ("rría", "rrías"), 0):
            continue
        # Length floor is just "at least one stem letter", not
        # _ES_MIN_STEM's usual +3/+4 margin: "ir" is a genuinely
        # complete, valid 2-letter infinitive, so its condicional
        # "iría" (4 letters total) is as short as this gets and would
        # otherwise be wrongly excluded by this section's usual
        # (longer-word-oriented) length guards.
        if _es_ends_with_any_in_verb_position(tokens, i, ("ría", "rías"), 4):
            return True
    return False


def _es_detect_perfecto_compuesto(sentence, tokens):
    haber_forms = {"he", "has", "ha", "hemos", "habéis", "han"}
    for i, t in enumerate(tokens):
        if t in haber_forms:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _es_ends_with_any(tokens[j], ("ado", "ido"), _ES_MIN_STEM + 3) or tokens[j] in (
                    "hecho", "dicho", "escrito", "visto", "puesto", "vuelto", "abierto", "cubierto",
                    "muerto", "roto", "resuelto", "impreso",
                ):
                    return True
    return False


def _es_detect_pluscuamperfecto(sentence, tokens):
    haber_forms = {"había", "habías", "habíamos", "habíais", "habían"}
    for i, t in enumerate(tokens):
        if t in haber_forms:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _es_ends_with_any(tokens[j], ("ado", "ido"), _ES_MIN_STEM + 3) or tokens[j] in (
                    "hecho", "dicho", "escrito", "visto", "puesto", "vuelto", "abierto", "cubierto",
                    "muerto", "roto", "resuelto", "impreso",
                ):
                    return True
    return False


def _es_detect_progresivo(sentence, tokens):
    for i, t in enumerate(tokens):
        if t in _ES_ESTAR_FORMS:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _es_ends_with_any(tokens[j], ("ando", "iendo", "yendo"), _ES_MIN_STEM + 4):
                    return True
    return False


# ---------------------------------------------------------------- mood

# Phrases that grammatically REQUIRE a subjunctive verb in the clause
# they introduce — a real, deterministic rule of Spanish grammar (not a
# guess about which of two possible verb classes a bare ending belongs
# to, the trap the tense endings themselves fall into: -ar subjunctive
# endings ("hable") are letter-for-letter identical to -er/-ir
# INDICATIVE endings, and vice versa, so there is no reliable ending-
# only signal — see this section's own docstring). Kept to phrases
# where the subjunctive is genuinely mandatory, not just common,
# excluding trigger words with a common non-subjunctive-requiring
# reading too (e.g. "cuando" only requires it for a FUTURE event, and
# "aunque" doesn't require it at all — both left out rather than risk a
# wrong claim).
_ES_SUBJUNTIVO_TRIGGERS = (
    "ojalá que", "ojalá", "espero que", "esperamos que", "quiero que", "quieres que", "queremos que",
    "es importante que", "es necesario que", "dudo que", "dudamos que", "no creo que", "no creemos que",
    "es posible que", "es probable que", "para que", "sin que", "antes de que",
)


def _es_detect_subjuntivo_presente(sentence, tokens):
    if any(t in _ES_SUBJUNTIVO_PRESENTE_IRREGULAR for t in tokens):
        return True
    padded = " " + " ".join(tokens) + " "
    for trigger in _ES_SUBJUNTIVO_TRIGGERS:
        marker = f" {trigger} "
        idx = padded.find(marker)
        if idx == -1:
            continue
        after = padded[idx + len(marker):].split()
        for w in after[:3]:
            if _es_ends_with_any(w, ("e", "es", "emos", "éis", "en", "a", "as", "amos", "áis", "an"), _ES_MIN_STEM):
                return True
    return False
    return False



# ser/ir share the single irregular preterito stem "fu-" (fui, fuiste,
# fue...), which carries through to their imperfect subjunctive too —
# "fuera"/"fuese", not the regularly-expected "fuiera"/"fuiese" the
# general -iera/-iese suffix check below would look for. Exact-matched
# here since it's common enough (this whole tense mostly exists to
# fill "si" clauses, and "fuera" — "were" — is one of the single most
# frequent verbs to fill one) to be worth naming directly rather than
# leaving as a gap the way a rarer irregular's own subjunctive-
# imperfect quirk is accepted to be.
_ES_SUBJUNTIVO_IMPERFECTO_IRREGULAR = {
    "fuera", "fueras", "fuéramos", "fuerais", "fueran",
    "fuese", "fueses", "fuésemos", "fueseis", "fuesen",
}


def _es_detect_subjuntivo_imperfecto(sentence, tokens):
    if any(t in _ES_SUBJUNTIVO_IMPERFECTO_IRREGULAR for t in tokens):
        return True
    for t in tokens:
        if _es_ends_with_any(
            t, ("áramos", "arais", "aran", "iéramos", "ierais", "ieran",
                "ásemos", "aseis", "asen", "iésemos", "ieseis", "iesen"),
            _ES_MIN_STEM + 5,
        ):
            return True
    for i, t in enumerate(tokens):
        if _es_pronoun_before(tokens, i) and _es_ends_with_any(
            t, ("ara", "aras", "iera", "ieras", "ase", "ases", "iese", "ieses"), _ES_MIN_STEM + 3,
        ):
            return True
    return False


_ES_ATTACHED_PRONOUN_SUFFIXES = ("melo", "mela", "melos", "melas", "te", "me", "nos",
                                  "selo", "sela", "selos", "selas", "se", "lo", "la", "los", "las", "le", "les")


def _es_detect_imperativo(sentence, tokens):
    if any(t in _ES_IMPERATIVO_IRREGULAR for t in tokens):
        return True
    # A written accent is the one reliable signal for the OTHER regular-
    # verb command shape this catches: an object/reflexive pronoun
    # attached directly onto the end of the verb ("cómelo", "háblame",
    # "díselo", "siéntate") — something that happens ONLY in commands
    # (and, without an accent, gerunds/infinitives — but attaching a
    # pronoun there doesn't shift the stress enough to ever need one,
    # so an ACCENTED form specifically is safe). Deliberately narrow:
    # a short command needing no accent to preserve its stress
    # ("dame", "ponlo", "vente") isn't caught by this and is accepted
    # as a real gap rather than guessed at — see this section's
    # docstring on why "no + tú-form" was tried and rejected as the
    # general rule here (identical spelling to plain negated statements
    # like "no dudas de mí", "you don't doubt me").
    for t in tokens:
        if any(a in t for a in "áéíóú") and _es_ends_with_any(t, _ES_ATTACHED_PRONOUN_SUFFIXES, _ES_MIN_STEM + 3):
            return True
    return False


# ----------------------------------------------------------- ser vs. estar

def _es_detect_ser(sentence, tokens):
    return any(t in _ES_SER_FORMS for t in tokens)


def _es_detect_estar(sentence, tokens):
    return any(t in _ES_ESTAR_FORMS for t in tokens)


# --------------------------------------------------- clauses & conditionals

def _es_si_index(tokens):
    for i, t in enumerate(tokens):
        if t == "si":
            return i
    return None


def _es_detect_clausula_que(sentence, tokens):
    for i, t in enumerate(tokens):
        if t == "que" and i > 0:
            return True
    return False


_ES_PRESENTE_SINGULAR_SUFFIXES = ("o", "as", "a", "es", "e")


def _es_has_presente_after(tokens, start, window=4):
    """Whether a presente-tense verb appears in tokens[start:start+window]
    — used for the word right after "si" in a type-1 conditional, a
    genuinely safer position than this section's general
    _es_detect_presente() singular check requires elsewhere: a
    conditional's "si" clause is short and its next content word is
    reliably that clause's own verb (with an optional subject noun/
    pronoun before it, not after), so the noun-precursor guard (see
    _es_ends_with_any_in_verb_position()) is enough here even without
    requiring an explicit pronoun — needed because Spanish drops the
    subject constantly ("si llueve", not "si ello llueve"). Plural
    endings still route through _es_classify_plural_ending() rather
    than a bare suffix check, for the same reason _es_detect_presente()
    itself does — otherwise "si hablaríamos" (conditional plural) would
    wrongly register a presente here too, via the same nested-ending
    coincidence described at _ES_PLURAL_ENDING_CLASSES.
    """
    end = min(start + window, len(tokens))
    for i in range(start, end):
        if tokens[i] in _ES_PRESENTE_IRREGULAR:
            return True
        if "presente" in _es_classify_plural_ending(tokens[i]):
            return True
        if _es_ends_with_any_in_verb_position(tokens, i, _ES_PRESENTE_SINGULAR_SUFFIXES, _ES_MIN_STEM):
            return True
    return False


def _es_detect_condicional_tipo1(sentence, tokens):
    idx = _es_si_index(tokens)
    if idx is None:
        return False
    return _es_has_presente_after(tokens, idx + 1) and _es_detect_futuro(sentence, tokens)


def _es_detect_condicional_tipo2(sentence, tokens):
    idx = _es_si_index(tokens)
    if idx is None:
        return False
    after = tokens[idx + 1:]
    has_imp_subj = any(
        _es_ends_with_any(t, ("ara", "aras", "áramos", "arais", "aran",
                              "iera", "ieras", "iéramos", "ierais", "ieran"), _ES_MIN_STEM + 3)
        for t in after
    )
    return has_imp_subj and _es_detect_condicional(sentence, tokens)


def _es_detect_condicional_tipo3(sentence, tokens):
    idx = _es_si_index(tokens)
    if idx is None:
        return False
    after = tokens[idx + 1:]
    hubiera_forms = {
        "hubiera", "hubieras", "hubiéramos", "hubierais", "hubieran",
        "hubiese", "hubieses", "hubiésemos", "hubieseis", "hubiesen",
    }
    has_pluscuamperfecto_subj = any(t in hubiera_forms for t in after)
    if not has_pluscuamperfecto_subj:
        return False
    habria_forms = {"habría", "habrías", "habríamos", "habríais", "habrían"}
    return any(t in habria_forms for t in tokens)


# ------------------------------------------------------------- pronouns

def _es_detect_a_personal(sentence, tokens):
    for i, t in enumerate(tokens):
        if t == "a" and i + 1 < len(tokens) and tokens[i + 1] not in ("el", "la", "los", "las", "quien", "quién"):
            # Deliberately narrow: only the clear, common shape "a" +
            # a capitalized proper name in the ORIGINAL sentence (a
            # person's name) — telling personal "a" apart from the
            # preposition "a" ("to"/"at", by far the more frequent use
            # of this same word) in front of an ordinary noun needs
            # knowing whether that noun refers to a specific person,
            # which word shape alone can't answer. Under-detects on
            # purpose rather than flagging every directional/temporal
            # "a" as if it were this construction.
            words = re.findall(_WORD_PATTERN, sentence, re.UNICODE)
            if i < len(words) - 1 and words[i + 1][:1].isupper():
                return True
    return False


def _es_detect_pronombre_objeto(sentence, tokens):
    return any(t in _ES_OBJECT_PRONOUNS for t in tokens)


def _es_detect_reflexivo(sentence, tokens):
    for i, t in enumerate(tokens):
        if t in _ES_REFLEXIVE_PRONOUNS:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _es_ends_with_any(tokens[j], ("o", "as", "a", "amos", "an", "ó", "aba", "e", "es"), _ES_MIN_STEM):
                    return True
    return False


# ----------------------------------------------------------- comparison

def _es_detect_comparativo(sentence, tokens):
    for i, t in enumerate(tokens):
        if t in ("más", "menos") and i + 2 < len(tokens) and tokens[i + 2] == "que":
            return True
        if t == "tan" and i + 2 < len(tokens) and tokens[i + 2] == "como":
            return True
    return False


def _es_detect_superlativo(sentence, tokens):
    for i, t in enumerate(tokens):
        if t in ("el", "la", "los", "las") and i + 1 < len(tokens) and tokens[i + 1] in ("más", "menos"):
            return True
        if _es_ends_with_any(t, ("ísimo", "ísima", "ísimos", "ísimas"), _ES_MIN_STEM + 5):
            return True
    return False


# ------------------------------------------------------- non-finite forms

def _es_detect_gerundio(sentence, tokens):
    for i, t in enumerate(tokens):
        if _es_ends_with_any(t, ("ando", "iendo", "yendo"), _ES_MIN_STEM + 4):
            if i > 0 and tokens[i - 1] in _ES_ESTAR_FORMS:
                continue  # already counted as progressive above
            return True
    return False


def _es_detect_infinitivo(sentence, tokens):
    verb_precedents = _ES_PRESENTE_IRREGULAR | {"quiero", "puedo", "debo", "voy", "tengo"}
    for i, t in enumerate(tokens):
        if t in ("a", "de", "para", "antes", "después", "sin", "que") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if _es_ends_with_any(nxt, ("ar", "er", "ir"), _ES_MIN_STEM + 2):
                return True
    for i, t in enumerate(tokens):
        if t in verb_precedents and i + 1 < len(tokens) and _es_ends_with_any(tokens[i + 1], ("ar", "er", "ir"), _ES_MIN_STEM + 2):
            return True
    return False


def _es_detect_participio_adjetivo(sentence, tokens):
    for i, t in enumerate(tokens):
        if t in _ES_SER_FORMS or t in _ES_ESTAR_FORMS:
            for j in range(i + 1, min(i + 3, len(tokens))):
                if _es_ends_with_any(tokens[j], ("ado", "ada", "ados", "adas", "ido", "ida", "idos", "idas"), _ES_MIN_STEM + 3):
                    return True
    return False


# --------------------------------------------------------------- questions

def _es_detect_pregunta_si_no(sentence, tokens):
    if "¿" not in sentence or "?" not in sentence:
        return False
    inner = sentence.split("¿", 1)[1].split("?", 1)[0].strip()
    inner_tokens = _es_tokenize(inner)
    if not inner_tokens or inner_tokens[0] in _ES_QUESTION_WORDS or inner_tokens[0] == "por":
        return False
    return True


def _es_detect_pregunta_qu(sentence, tokens):
    if "¿" not in sentence:
        return False
    inner = sentence.split("¿", 1)[1]
    inner_tokens = _es_tokenize(inner)
    if not inner_tokens:
        return False
    if inner_tokens[0] in _ES_QUESTION_WORDS:
        return True
    return inner_tokens[0] == "por" and len(inner_tokens) > 1 and inner_tokens[1] == "qué"


# ----------------------------------------------------------------- catalog

CATEGORY_ORDER_ES = (
    "Tiempos Verbales", "Modo", "Ser vs. Estar", "Cláusulas y Condicionales",
    "Pronombres", "Comparación", "Formas No Personales", "Preguntas",
)

_RULES_ES = [
    # ---- Tiempos Verbales (Tenses & Aspect) ----
    {
        "id": "es_presente", "category": "Tiempos Verbales", "name": "Presente",
        "definition": "Used for habits, routines, general truths, and things happening right now.",
        "rule_explanation": "Regular -ar/-er/-ir endings (hablo, como, vivo...), or a subject pronoun + verb: \"Ella habla español.\"",
        "detect": _es_detect_presente,
    },
    {
        "id": "es_preterito", "category": "Tiempos Verbales", "name": "Pretérito Indefinido",
        "definition": "Used for a completed action or event in the past, viewed as a finished whole.",
        "rule_explanation": "Regular endings -é/-aste/-ó/-amos/-asteis/-aron (-ar) or -í/-iste/-ió/-imos/-isteis/-ieron (-er/-ir): \"Ella habló con su madre.\"",
        "detect": _es_detect_preterito,
    },
    {
        "id": "es_imperfecto", "category": "Tiempos Verbales", "name": "Pretérito Imperfecto",
        "definition": "Used for ongoing or repeated past actions, background description, and past habits — no fixed endpoint in mind.",
        "rule_explanation": "Endings -aba/-abas/-aba/-ábamos/-abais/-aban (-ar) or -ía/-ías/-ía/-íamos/-íais/-ían (-er/-ir): \"Ella hablaba todos los días.\"",
        "detect": _es_detect_imperfecto,
    },
    {
        "id": "es_futuro", "category": "Tiempos Verbales", "name": "Futuro Simple",
        "definition": "Used for future actions, predictions, and (informally) guesses about the present.",
        "rule_explanation": "Infinitive (or an irregular future stem) + -é/-ás/-á/-emos/-éis/-án: \"Ella hablará mañana.\"",
        "detect": _es_detect_futuro,
    },
    {
        "id": "es_condicional", "category": "Tiempos Verbales", "name": "Condicional Simple",
        "definition": "Used for hypothetical actions (\"would\"), polite requests, and the result clause of an unreal condition.",
        "rule_explanation": "Infinitive (or an irregular future stem) + -ía/-ías/-ía/-íamos/-íais/-ían: \"Ella hablaría contigo.\"",
        "detect": _es_detect_condicional,
    },
    {
        "id": "es_perfecto_compuesto", "category": "Tiempos Verbales", "name": "Pretérito Perfecto Compuesto",
        "definition": "Used for a past action with a present connection — a result that still matters now (like English's present perfect).",
        "rule_explanation": "he/has/ha/hemos/habéis/han + past participle (-ado/-ido, or an irregular one like \"hecho\"): \"Ella ha hablado con él.\"",
        "detect": _es_detect_perfecto_compuesto,
    },
    {
        "id": "es_pluscuamperfecto", "category": "Tiempos Verbales", "name": "Pluscuamperfecto",
        "definition": "Used for an action that had already happened before another past action or time.",
        "rule_explanation": "había/habías/habíamos/habíais/habían + past participle: \"Ella ya había hablado cuando llegamos.\"",
        "detect": _es_detect_pluscuamperfecto,
    },
    {
        "id": "es_progresivo", "category": "Tiempos Verbales", "name": "Estar + Gerundio (Progresivo)",
        "definition": "Used for an action in progress right at the moment being described.",
        "rule_explanation": "A form of estar + gerund (-ando/-iendo): \"Ella está hablando por teléfono.\"",
        "detect": _es_detect_progresivo,
    },
    # ---- Modo (Mood) ----
    {
        "id": "es_subjuntivo_presente", "category": "Modo", "name": "Presente de Subjuntivo",
        "definition": "Used (instead of the indicative) after expressions of wish, doubt, emotion, necessity, or purpose — the clause describes something not asserted as simple fact.",
        "rule_explanation": (
            "A trigger like \"espero que\", \"quiero que\", \"es necesario que\", \"para que\" + a "
            "subjunctive-shaped verb: \"Espero que ella venga.\" Note: only detected after one of these "
            "specific, genuinely subjunctive-requiring triggers — the endings themselves (\"hable\", \"coma\") "
            "are letter-for-letter identical to a DIFFERENT verb class's ordinary indicative endings, so there's "
            "no reliable way to recognize this mood from spelling alone without a trigger alongside it."
        ),
        "detect": _es_detect_subjuntivo_presente,
    },
    {
        "id": "es_subjuntivo_imperfecto", "category": "Modo", "name": "Imperfecto de Subjuntivo",
        "definition": "The past subjunctive — used after the same triggers as the present subjunctive (wish, doubt, emotion, unreal conditions) but referring to the past, or in a type-2 \"if\" clause.",
        "rule_explanation": "Endings -ara/-aras/-áramos/-arais/-aran or -iera/-ieras/-iéramos/-ierais/-ieran (also the equally correct -ase/-iese forms): \"Si ella hablara español...\"",
        "detect": _es_detect_subjuntivo_imperfecto,
    },
    {
        "id": "es_imperativo", "category": "Modo", "name": "Imperativo",
        "definition": "Commands — telling someone to do (or not do) something directly.",
        "rule_explanation": "One of the 8 irregular tú-commands (sé, ve, ten, haz, pon, sal, ven, di), or \"no\" + a subjunctive-shaped verb for a negative command: \"¡Ven aquí!\" / \"No hables tan rápido.\"",
        "detect": _es_detect_imperativo,
    },
    # ---- Ser vs. Estar ----
    {
        "id": "es_ser", "category": "Ser vs. Estar", "name": "Ser",
        "definition": "Used for identity, characteristics, origin, material, time, and other things treated as inherent or permanent.",
        "rule_explanation": "soy/eres/es/somos/sois/son (and its other tenses): \"Ella es doctora.\"",
        "detect": _es_detect_ser,
    },
    {
        "id": "es_estar", "category": "Ser vs. Estar", "name": "Estar",
        "definition": "Used for location, temporary states/conditions, and (with a gerund) ongoing actions.",
        "rule_explanation": "estoy/estás/está/estamos/estáis/están (and its other tenses): \"Ella está cansada.\"",
        "detect": _es_detect_estar,
    },
    # ---- Cláusulas y Condicionales ----
    {
        "id": "es_clausula_que", "category": "Cláusulas y Condicionales", "name": "Cláusulas con \"que\"",
        "definition": "A relative or noun clause introduced by \"que\" — describing a noun, or acting as the object of a verb (much like English \"that\"/\"which\").",
        "rule_explanation": "... + que + clause: \"Creo que ella tiene razón.\" / \"El libro que compré es interesante.\"",
        "detect": _es_detect_clausula_que,
    },
    {
        "id": "es_condicional_tipo1", "category": "Cláusulas y Condicionales", "name": "Condicional Tipo 1 (Real)",
        "definition": "The \"real/possible\" conditional — a realistic present condition and its likely future result.",
        "rule_explanation": "Si + presente, ... futuro: \"Si llueve, me quedaré en casa.\"",
        "detect": _es_detect_condicional_tipo1,
    },
    {
        "id": "es_condicional_tipo2", "category": "Cláusulas y Condicionales", "name": "Condicional Tipo 2 (Irreal Presente)",
        "definition": "The \"unreal present\" conditional — an imaginary or unlikely present/future situation and its imagined result.",
        "rule_explanation": "Si + imperfecto de subjuntivo, ... condicional: \"Si tuviera más tiempo, viajaría.\"",
        "detect": _es_detect_condicional_tipo2,
    },
    {
        "id": "es_condicional_tipo3", "category": "Cláusulas y Condicionales", "name": "Condicional Tipo 3 (Irreal Pasado)",
        "definition": "The \"unreal past\" conditional — an imaginary past situation and how things would have turned out differently.",
        "rule_explanation": "Si + pluscuamperfecto de subjuntivo, ... condicional compuesto: \"Si hubiera estudiado, habría aprobado.\"",
        "detect": _es_detect_condicional_tipo3,
    },
    # ---- Pronombres ----
    {
        "id": "es_a_personal", "category": "Pronombres", "name": "La \"a\" Personal",
        "definition": "A required \"a\" placed before a direct object that's a specific person — has no equivalent in English.",
        "rule_explanation": "Verb + a + [person's name]: \"Vi a María en el parque.\" Note: only detected before a capitalized name, to avoid confusing this with the ordinary preposition \"a\" (\"to\"/\"at\"), by far its more common use.",
        "detect": _es_detect_a_personal,
    },
    {
        "id": "es_pronombre_objeto", "category": "Pronombres", "name": "Pronombres de Objeto",
        "definition": "Direct/indirect object pronouns — replace a noun already mentioned, and attach before a conjugated verb (or after an infinitive/gerund/command).",
        "rule_explanation": "lo/la/los/las (direct object) or le/les (indirect object) + verb: \"Lo vi ayer.\" / \"Le di el libro.\"",
        "detect": _es_detect_pronombre_objeto,
    },
    {
        "id": "es_reflexivo", "category": "Pronombres", "name": "Verbos Reflexivos",
        "definition": "Used when the subject does the action to itself — the pronoun and the subject are the same person.",
        "rule_explanation": "me/te/se/nos/os + verb: \"Ella se levanta temprano.\"",
        "detect": _es_detect_reflexivo,
    },
    # ---- Comparación ----
    {
        "id": "es_comparativo", "category": "Comparación", "name": "Comparativos",
        "definition": "Used to compare two things.",
        "rule_explanation": "más/menos + adjective/adverb + que, or tan + adjective/adverb + como: \"Ella es más alta que su hermano.\"",
        "detect": _es_detect_comparativo,
    },
    {
        "id": "es_superlativo", "category": "Comparación", "name": "Superlativos",
        "definition": "Used to say something is the highest or lowest degree among a group, or simply \"very [adjective]\".",
        "rule_explanation": "el/la/los/las + más/menos + adjective, or adjective + -ísimo/-ísima: \"Es el más inteligente de la clase.\" / \"Es facilísimo.\"",
        "detect": _es_detect_superlativo,
    },
    # ---- Formas No Personales (Non-finite Forms) ----
    {
        "id": "es_gerundio", "category": "Formas No Personales", "name": "Gerundio",
        "definition": "The \"-ing\"-equivalent form, used adverbially (describing how/when something is done) — not with \"estar\" here (see Progresivo).",
        "rule_explanation": "-ando (-ar) or -iendo/-yendo (-er/-ir): \"Caminando por la calle, vio a su amigo.\"",
        "detect": _es_detect_gerundio,
    },
    {
        "id": "es_infinitivo", "category": "Formas No Personales", "name": "Infinitivo",
        "definition": "The unconjugated \"to [verb]\" form, used after another verb, after a preposition, or as a verbal noun.",
        "rule_explanation": "a/de/para/antes de/después de/sin + infinitive, or verb + infinitive: \"Quiero comer.\" / \"Antes de salir, cerró la puerta.\"",
        "detect": _es_detect_infinitivo,
    },
    {
        "id": "es_participio_adjetivo", "category": "Formas No Personales", "name": "Participio como Adjetivo",
        "definition": "A past participle used as a descriptive adjective, agreeing in gender/number with the noun it describes.",
        "rule_explanation": "ser/estar + -ado/-ada/-ados/-adas or -ido/-ida/-idos/-idas: \"La puerta está cerrada.\"",
        "detect": _es_detect_participio_adjetivo,
    },
    # ---- Preguntas ----
    {
        "id": "es_pregunta_si_no", "category": "Preguntas", "name": "Preguntas de Sí/No",
        "definition": "A question that can be answered with \"sí\" or \"no\", not starting with a question word.",
        "rule_explanation": "¿ + [no question word] ... ?: \"¿Hablas español?\"",
        "detect": _es_detect_pregunta_si_no,
    },
    {
        "id": "es_pregunta_qu", "category": "Preguntas", "name": "Preguntas con Palabra Interrogativa",
        "definition": "A question asking for specific information, starting with a question word (always written with an accent).",
        "rule_explanation": "¿Qué/quién/cómo/cuándo/dónde/por qué/cuál/cuánto...?: \"¿Dónde vives?\"",
        "detect": _es_detect_pregunta_qu,
    },
]


def analyze_grammar(sentences, lang="en", should_continue=None):
    """{rule_id: {"name", "category", "definition", "rule_explanation",
    "sentences", "total_count"}} for every rule the catalog above found
    at least one match for in `sentences` — rules with zero matches are
    simply absent, since the app shows grammar "used in the book," not
    the full catalog. English and Spanish only: any other `lang`
    returns {} immediately, same short-circuit convention as
    text_analyzer.build_cefr_map() — English via nltk's POS-tag-sequence
    rules (_RULES), Spanish via the suffix/lexical-anchor rules in the
    "Spanish" section above (_RULES_ES), a genuinely different detection
    strategy for a genuinely different reason (see that section's own
    docstring) even though both feed the exact same result shape and
    the exact same UI below.

    `sentences` should be the already-split list text_analyzer's
    split_sentences() produced for this book (the app's own cached
    self.book_sentences) — not re-split here.

    `should_continue`, if given, is polled between sentences exactly
    like build_glossary()'s should_continue: when it returns False the
    loop stops and whatever was accumulated so far is returned, not
    discarded.

    "sentences" inside each result is capped at _MAX_SENTENCES_PER_RULE
    (matching concordance.py's own capped-results convention);
    "total_count" is the true, uncapped count, so the UI can show
    "showing first 300 of N" the same way ConcordanceWindow already
    does for search results.
    """
    if lang == "en":
        rules = _RULES
    elif lang == "es":
        rules = _RULES_ES
    else:
        return {}
    results = {}
    for sentence in sentences:
        if should_continue is not None and not should_continue():
            break
        sentence = sentence.strip()
        if not sentence or len(sentence) > _MAX_SENTENCE_LEN:
            continue
        if lang == "en":
            match_args = (nltk.pos_tag(nltk.word_tokenize(sentence)),)
        else:
            match_args = (sentence, _es_tokenize(sentence))
        for rule in rules:
            if not rule["detect"](*match_args):
                continue
            bucket = results.setdefault(rule["id"], {
                "name": rule["name"],
                "category": rule["category"],
                "definition": rule["definition"],
                "rule_explanation": rule["rule_explanation"],
                "sentences": [],
                "total_count": 0,
            })
            bucket["total_count"] += 1
            if len(bucket["sentences"]) < _MAX_SENTENCES_PER_RULE:
                bucket["sentences"].append(sentence)
    return results


# ----------------------------------------------------------------- UI

import random
import tkinter as tk
from tkinter import ttk

from text_analyzer import due_words, mark_known, review_word

import theme as th


def build_grammar_section(parent, palette, themed, on_select_rule, on_mark):
    """Builds Section 2's own master-detail layout: a category/rule
    Treeview on the left, a scrollable definition+examples panel on the
    right — a left/right split, unlike the vocabulary section's
    top/bottom split, since a rule's matching sentences need real
    vertical room (potentially dozens of them), not a 4-line blurb.

    `on_select_rule` is called with the selected rule's id (or None,
    when the selection is cleared) — app.py owns the actual data
    (self.grammar_results) and populates the detail widgets itself, the
    same division of responsibility render_results()/_show_selection()
    already use for the vocabulary Treeview. `on_mark` is called with
    "known"/"learning" from the two buttons below the detail text,
    mirroring the vocab detail panel's detail_know_btn/
    detail_learning_btn — app.py owns updating grammar_known/
    grammar_learning and re-rendering afterward, same division again.

    Returns (frame, widgets) — widgets = {"tree", "detail_header_var",
    "detail_text", "know_btn", "learning_btn"} — same (frame,
    widgets)-dict convention as study.build_study_summary_card().
    """
    p = palette
    frame = tk.Frame(parent, bg=p["BG"])

    # Wide enough for the longest real rule name ("Present Perfect
    # Continuous", "Present Participle Phrases", 26 characters) plus its
    # tree-hierarchy indent (a rule row sits one level under its category
    # row, which eats into the "#0" column's usable text width beyond
    # the column's own nominal size) without clipping — confirmed
    # directly: 230px/340px were both still cutting names off
    # mid-word, not just the header label this comment used to be about.
    list_frame = tk.Frame(frame, bg=p["BG"], width=400)
    list_frame.pack(side="left", fill="y", padx=(0, 10))
    list_frame.pack_propagate(False)
    themed.add(list_frame, bg="BG")

    # Shorter text plus a wraplength safety net (wrapping to a second
    # line reads fine; silent clipping doesn't) rather than relying on
    # width tuning alone to always be enough at every DPI setting.
    header = tk.Label(
        list_frame, text="GRAMMAR STRUCTURES", bg=p["BG"], fg=p["FG"], font=th.FONT_BOLD,
        wraplength=380, justify="left", anchor="w",
    )
    header.pack(anchor="w", fill="x", pady=(0, 6))
    themed.add(header, bg="BG", fg="FG")

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
    tree.heading("#0", text="Structure")
    # "Count", not "Sentences" — matches the vocabulary tree's own count
    # column header exactly, and comfortably fits this narrower column.
    tree.heading("count", text="Count")
    tree.column("#0", width=320, anchor="w", stretch=False)
    tree.column("count", width=60, anchor="center", stretch=False)
    tree.tag_configure("category", foreground=p["ACCENT"], font=th.FONT_SMALL_BOLD)
    # Same status-color roles the vocabulary tree's "new"/"learning"/
    # "known" tags already use, applied here too now that grammar rules
    # get the same know/learning marking.
    tree.tag_configure("new", foreground=p["ACCENT"], font=th.FONT_ROW_WORD)
    tree.tag_configure("learning", foreground=p["SECOND_FG"], font=th.FONT_ROW_WORD)
    tree.tag_configure("known", foreground=p["DIM"], font=th.FONT)

    detail_frame = tk.Frame(frame, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
    detail_frame.pack(side="left", fill="both", expand=True)
    themed.add(detail_frame, bg="PANEL", highlightbackground="PANEL_BORDER")

    detail_header_var = tk.StringVar(value="Select a grammar structure to see details")
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
    detail_text.tag_configure("rule", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.tag_configure("num", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.tag_configure("example", foreground=p["RUBRIC"], font=th.FONT)
    detail_text.tag_configure("hint", foreground=p["DIM"], font=th.FONT_SMALL_ITALIC)
    detail_text.config(state="disabled")
    themed.add(detail_text, bg="PANEL", fg="FG", insertbackground="FG")

    btn_row = tk.Frame(detail_frame, bg=p["PANEL"])
    btn_row.pack(fill="x", padx=14, pady=(0, 10))
    themed.add(btn_row, bg="PANEL")
    know_btn = ttk.Button(
        btn_row, text="✅ I KNOW IT", command=lambda: on_mark("known"), state="disabled",
    )
    know_btn.pack(side="left")
    learning_btn = ttk.Button(
        btn_row, text="\U0001F4D6 STILL LEARNING", command=lambda: on_mark("learning"), state="disabled",
    )
    learning_btn.pack(side="left", padx=(8, 0))

    def _on_select(_event):
        sel = tree.selection()
        on_select_rule(sel[0] if sel else None)

    tree.bind("<<TreeviewSelect>>", _on_select)

    return frame, {
        "tree": tree, "detail_header_var": detail_header_var, "detail_text": detail_text,
        "know_btn": know_btn, "learning_btn": learning_btn,
    }


def build_sentence_index(grammar_results, max_sentences=40):
    """{sentence: [(rule_id, data), ...]} for the "Identify Grammar" quiz
    mode below — every sentence that appears in at least one rule's
    (already-capped) sentence list, mapped to EVERY rule it satisfies.
    A sentence commonly matches several rules at once (a past-simple
    passive sentence is both "Past Simple" and "Passive Voice"), and the
    quiz card needs the full set to grade a self-reported answer fairly
    — a reader who correctly names only one of two real answers
    shouldn't be treated as having missed the card entirely, but they
    also shouldn't be quizzed on a version of the question that hides
    the fact that more than one answer exists (see reveal()'s "This
    sentence uses:" bullet list, not a single name).

    Capped at `max_sentences` via random sampling, not the first N
    found, so a long book's quiz session covers a representative mix of
    categories rather than whatever this book's iteration order
    happened to list first — matches the same "sample, don't just
    truncate" reasoning study.StudyWindow's own due/later shuffle uses.
    """
    by_sentence = {}
    for rule_id, data in grammar_results.items():
        for sentence in data["sentences"]:
            by_sentence.setdefault(sentence, []).append((rule_id, data))
    sentences = list(by_sentence.items())
    if len(sentences) > max_sentences:
        sentences = random.sample(sentences, max_sentences)
    return dict(sentences)


class GrammarQuizWindow(tk.Toplevel):
    """A simpler sibling to study.StudyWindow, for grammar structures
    instead of vocabulary words, with two modes:

    - "structure" (default): shows a rule's NAME, reveal shows its
      definition/rule/example sentences — direct recall, like a
      vocabulary flashcard.
    - "identify": shows a SENTENCE from the book, reveal shows every
      grammar structure that sentence actually uses (there can be more
      than one — see build_sentence_index()) — recognizing grammar in
      context, the reverse direction from "structure" mode.

    No writing mode (StudyWindow's typed-answer-against-one-exact-word
    grading, and masking that word out of an example sentence, have no
    equivalent for a grammar rule or a whole illustrative sentence), and
    no SPEAK button (pronunciation doesn't apply to either a rule name
    or a full sentence). Everything else generic — due-first ordering,
    SM-2 scheduling via review_word()/mark_known(), the
    on_finish(known_ids, learning_ids) contract, the summary screen —
    mirrors StudyWindow exactly, reusing the same widget/layout choices
    (readonly Entry + Text for real copy/select, three button rows to
    avoid the narrow-window clipping StudyWindow's own history already
    ran into) rather than inventing new ones.

    `cards` is always [(key, payload), ...]: in "structure" mode `key`
    is a rule_id and `payload` is that rule's data dict; in "identify"
    mode `key` is a sentence and `payload` is the list of (rule_id,
    data) tuples build_sentence_index() found for it — the caller
    (app.py's open_grammar_quiz()) builds whichever shape matches
    `mode`, the same division of responsibility every other window here
    uses (the window renders what it's handed, the app decides what
    that is).
    """

    def __init__(self, master, cards, on_finish=None, palette=None, schedule=None,
                 already_known_rules=None, learned_at=None, mode="structure"):
        super().__init__(master)
        self.p = palette or th.LIGHT
        self.mode = mode if mode in ("structure", "identify") else "structure"
        self.title("Grammar Quiz — Text Analyzer")
        self.geometry("640x520")
        self.minsize(560, 420)
        self.configure(bg=self.p["BG"])
        self.on_finish = on_finish
        self.schedule = schedule
        self.learned_at = learned_at
        self.already_known_rules = already_known_rules if already_known_rules is not None else set()

        def _rule_ids_of(card):
            key, payload = card
            return [key] if self.mode == "structure" else [rid for rid, _d in payload]

        cards = list(cards)
        if self.schedule is not None:
            due_rule_ids = set(due_words(self.schedule, sorted({
                rid for card in cards for rid in _rule_ids_of(card)
            })))
            due_entries = [c for c in cards if any(rid in due_rule_ids for rid in _rule_ids_of(c))]
            later_entries = [c for c in cards if c not in due_entries]
        else:
            due_entries, later_entries = cards, []
        random.shuffle(due_entries)
        random.shuffle(later_entries)
        self.cards = due_entries + later_entries
        self._rule_ids_of = _rule_ids_of
        self.index = 0
        self.known_rules = []      # [(rule_id, name), ...] — deduped in show_summary()
        self.learning_rules = []   # [(rule_id, name), ...]
        self.revealed = False
        self.finished = False

        self.counter_label = tk.Label(self, text="", bg=self.p["BG"], fg=self.p["DIM"], font=th.FONT)
        self.counter_label.pack(pady=(20, 0))

        self.card_frame = tk.Frame(
            self, bg=self.p["PANEL"], highlightthickness=1, highlightbackground=self.p["PANEL_BORDER"],
        )
        self.card_frame.pack(fill="both", expand=True, padx=30, pady=20)

        self.name_var = tk.StringVar()
        self.name_entry = tk.Entry(
            self.card_frame, textvariable=self.name_var, state="readonly",
            readonlybackground=self.p["PANEL"], fg=self.p["ACCENT"], font=th.FONT_CARD_WORD,
            relief="flat", highlightthickness=0, bd=0, justify="center", cursor="xterm",
            insertbackground=self.p["ACCENT"], selectbackground=self.p["ACCENT"],
            selectforeground=self.p["ACCENT_TEXT"],
        )

        self.card_text = tk.Text(
            self.card_frame, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word",
            relief="flat", highlightthickness=0, padx=0, pady=0, borderwidth=0,
            cursor="xterm", insertbackground=self.p["FG"], height=9,
        )
        self.card_text.tag_configure("def", foreground=self.p["FG"], font=th.FONT, justify="center")
        self.card_text.tag_configure("rule", foreground=self.p["DIM"], font=th.FONT_SMALL_ITALIC, justify="center")
        self.card_text.tag_configure("example", foreground=self.p["RUBRIC"], font=th.FONT_SMALL_ITALIC, justify="center")
        # "identify" mode's prompt (the sentence itself, shown before
        # reveal, in place of "structure" mode's name_entry) and the
        # bold rule-name lead-in each reveal() bullet starts with.
        self.card_text.tag_configure("prompt", foreground=self.p["ACCENT"], font=th.FONT_ROW_WORD, justify="center")
        self.card_text.tag_configure("answer_name", foreground=self.p["ACCENT"], font=th.FONT_BOLD, justify="center")
        self.card_text.config(state="disabled")

        self.summary_frame = tk.Frame(self.card_frame, bg=self.p["PANEL"])
        summary_scroll = tk.Scrollbar(
            self.summary_frame, bg=self.p["PANEL"], troughcolor=self.p["BG"],
            activebackground=self.p["SECOND_HOVER"], highlightthickness=0,
        )
        summary_scroll.pack(side="right", fill="y")
        self.summary_text = tk.Text(
            self.summary_frame, bg=self.p["PANEL"], fg=self.p["FG"], font=th.FONT, wrap="word",
            relief="flat", yscrollcommand=summary_scroll.set, highlightthickness=0, padx=10, pady=10,
        )
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scroll.config(command=self.summary_text.yview)
        self.summary_text.tag_configure("heading", font=th.FONT_ROW_WORD, foreground=self.p["ACCENT"])
        self.summary_text.tag_configure("known", foreground=self.p["FG"], font=th.FONT)
        self.summary_text.tag_configure("learning", foreground=self.p["RUBRIC"], font=th.FONT)
        self.summary_text.config(state="disabled")

        self.reveal_btn = ttk.Button(
            self, text="SHOW ANSWER (space)", command=self.reveal, style="Accent.TButton",
        )
        self.reveal_btn.pack(fill="x", padx=30, pady=(0, 8))

        self.verdict_row = tk.Frame(self, bg=self.p["BG"])
        self.learning_btn = ttk.Button(
            self.verdict_row, text="STILL LEARNING (2)", command=lambda: self.next_card(False),
        )
        self.learning_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.known_btn = ttk.Button(
            self.verdict_row, text="I KNOW IT (1)", command=lambda: self.next_card(True), style="Accent.TButton",
        )
        self.known_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.verdict_row.pack(fill="x", padx=30, pady=(0, 8))

        utility_row = tk.Frame(self, bg=self.p["BG"])
        utility_row.pack(fill="x", padx=30, pady=(0, 20))
        self.end_btn = ttk.Button(utility_row, text="END SESSION", command=self.end_session)
        self.end_btn.pack(side="right")

        self.bind("<space>", lambda _e: self.reveal())
        self.bind("<Return>", lambda _e: self.next_card(True))
        self.bind("<Key-1>", lambda _e: self.next_card(True))
        self.bind("<Key-2>", lambda _e: self.next_card(False))
        th.apply_dark_titlebar(self, self.p is th.DARK)
        self.show_card()
        self.focus_set()

    def show_card(self):
        # A card is skipped once every rule it involves is already known
        # — for a "structure" card that's just its one rule_id; for an
        # "identify" card (possibly several rules at once), ALL of them
        # need to already be known before the sentence itself stops
        # being worth asking about.
        while self.index < len(self.cards) and all(
            rid in self.already_known_rules for rid in self._rule_ids_of(self.cards[self.index])
        ):
            del self.cards[self.index]
        if self.index >= len(self.cards):
            self.show_summary()
            return
        self.summary_frame.pack_forget()
        self.card_text.pack(pady=10, padx=20, fill="x")
        key, _payload = self.cards[self.index]
        self.card_text.config(state="normal")
        self.card_text.delete("1.0", "end")
        if self.mode == "identify":
            # The sentence itself IS the prompt — shown up front, unlike
            # "structure" mode where the name is the starting point and
            # the sentence(s) only appear after reveal().
            self.name_entry.pack_forget()
            self.card_text.insert("end", f"“{key}”", "prompt")
            self.name_var.set("")
        else:
            self.name_entry.pack(pady=(40, 10))
            self.name_var.set(key)
        self.card_text.config(state="disabled")
        self.revealed = False
        self.reveal_btn.config(state="normal")
        self.counter_label.config(text=f"{self.index + 1}/{len(self.cards)}")

    def reveal(self):
        if self.finished or self.revealed:
            return
        _key, payload = self.cards[self.index]
        self.card_text.config(state="normal")
        if self.mode == "identify":
            # payload is [(rule_id, data), ...] — could be more than one
            # real answer for the same sentence (see
            # build_sentence_index()'s docstring), so every match is
            # listed, not just the first.
            self.card_text.insert("end", "\n\nThis sentence uses:\n", "rule")
            for _rid, data in payload:
                self.card_text.insert("end", f"\n{data['name']}", "answer_name")
                self.card_text.insert("end", f" — {data['definition']}", "def")
        else:
            # payload is a single rule's data dict.
            self.card_text.insert("end", payload["definition"], "def")
            self.card_text.insert("end", "\n\n" + payload["rule_explanation"], "rule")
            # Up to 3 illustrative sentences, not the full (possibly
            # hundreds-long) list — a quiz card is meant to jog recall,
            # not reproduce the whole detail-panel listing.
            for sentence in payload["sentences"][:3]:
                self.card_text.insert("end", f"\n\n“{sentence}”", "example")
        self.card_text.config(state="disabled")
        self.revealed = True

    def next_card(self, known):
        if self.finished:
            return
        for rule_id, name in self._card_rule_names(self.index):
            (self.known_rules if known else self.learning_rules).append((rule_id, name))
            if self.schedule is not None:
                review_word(self.schedule, rule_id, known)
            if known and self.learned_at is not None:
                mark_known(self.learned_at, rule_id)
        self.index += 1
        self.show_card()

    def _card_rule_names(self, index):
        """[(rule_id, name), ...] for cards[index] — one entry for a
        "structure" card, possibly several for an "identify" card whose
        sentence matched more than one rule. Marking such a card known
        marks every one of its rules known: recognizing all the grammar
        in a sentence is the whole point of that mode's verdict.
        """
        key, payload = self.cards[index]
        if self.mode == "identify":
            return [(rid, data["name"]) for rid, data in payload]
        return [(key, payload["name"])]

    def end_session(self):
        if not self.finished:
            self.show_summary()

    def show_summary(self):
        self.finished = True
        self.name_entry.pack_forget()
        self.card_text.pack_forget()
        self.summary_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Dedupe by rule_id: "identify" mode can mark the SAME rule
        # known/learning more than once (it appeared in several
        # sentence-cards this session) — a rule marked known on any card
        # wins over a "still learning" verdict from a different card in
        # the same session, same "known wins" precedent
        # app.py's _record_grammar_quiz_results() already applies when
        # merging a session's results into long-term progress.
        names_by_id = dict(self.known_rules) | dict(self.learning_rules)
        known_ids = {rid for rid, _name in self.known_rules}
        learning_ids = {rid for rid, _name in self.learning_rules} - known_ids

        if self.on_finish:
            self.on_finish(sorted(known_ids), sorted(learning_ids))

        reviewed = len(self.known_rules) + len(self.learning_rules)
        not_reached = len(self.cards) - reviewed
        self.counter_label.config(text=f"{reviewed}/{len(self.cards)} reviewed")
        self.reveal_btn.config(state="disabled")
        self.learning_btn.config(state="disabled")
        self.known_btn.config(state="disabled")
        self.end_btn.config(text="CLOSE", command=self.destroy)

        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        header = f"SESSION SUMMARY — {reviewed} reviewed"
        if not_reached:
            header += f", {not_reached} not reached"
        self.summary_text.insert("end", header + "\n\n", "heading")

        self.summary_text.insert("end", f"I KNOW IT  ({len(known_ids)})\n", "heading")
        if known_ids:
            for rid in sorted(known_ids, key=lambda r: names_by_id[r]):
                self.summary_text.insert("end", f"  {names_by_id[rid]}\n", "known")
        else:
            self.summary_text.insert("end", "  (none)\n", "known")

        self.summary_text.insert("end", "\n")
        self.summary_text.insert("end", f"STILL LEARNING  ({len(learning_ids)})\n", "heading")
        if learning_ids:
            for rid in sorted(learning_ids, key=lambda r: names_by_id[r]):
                self.summary_text.insert("end", f"  {names_by_id[rid]}\n", "learning")
        else:
            self.summary_text.insert("end", "  (none)\n", "learning")
        self.summary_text.config(state="disabled")


def build_grammar_readiness_card(parent, palette, on_start, on_export_anki, themed=None):
    """The docked "grammar readiness" card for the main window's right
    column when Section 2 is active — mirrors
    study.build_study_summary_card()'s shape and wording, two buttons
    into GrammarQuizWindow (one per `mode`, same "one button per mode"
    pattern build_study_summary_card()'s flashcard/writing pair already
    uses) plus one into Anki export.

    `on_start` is called with a single "structure"/"identify" argument
    (which GrammarQuizWindow mode to open), mirroring
    build_study_summary_card()'s own `on_start("flashcard"/"writing")`
    contract. `on_export_anki` takes none, same as its vocab sibling.

    Returns (frame, widgets) — widgets: book_label, meta_label,
    readiness_label, progress_bar, known_label, to_learn_label,
    count_label, quiz_btn, identify_btn, anki_btn.
    """
    p = palette
    frame = tk.Frame(parent, bg=p["PANEL"], highlightthickness=1, highlightbackground=p["PANEL_BORDER"])
    header = tk.Label(frame, text="GRAMMAR READINESS", bg=p["PANEL"], fg=p["FG"], font=th.FONT_BOLD)
    header.pack(anchor="w", padx=12, pady=(10, 0))

    book_label = tk.Label(
        frame, text="No book loaded yet.", bg=p["PANEL"], fg=p["FG"], font=th.FONT_SMALL_BOLD,
        wraplength=260, justify="left", anchor="w",
    )
    book_label.pack(anchor="w", padx=12, pady=(6, 0), fill="x")
    meta_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["DIM"], font=th.FONT_SMALL_ITALIC, anchor="w")
    meta_label.pack(anchor="w", padx=12, pady=(0, 8))

    readiness_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["ACCENT"], font=th.FONT_SMALL_BOLD, anchor="w")
    readiness_label.pack(anchor="w", padx=12)
    progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=100, value=0)
    progress_bar.pack(fill="x", padx=12, pady=(4, 8))

    known_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["FG"], font=th.FONT, anchor="w")
    known_label.pack(anchor="w", padx=12)
    to_learn_label = tk.Label(frame, text="", bg=p["PANEL"], fg=p["FG"], font=th.FONT, anchor="w")
    to_learn_label.pack(anchor="w", padx=12, pady=(0, 10))

    count_label = tk.Label(
        frame, text="0 structure(s) ready to study", bg=p["PANEL"], fg=p["DIM"], font=th.FONT,
    )
    count_label.pack(anchor="w", padx=12, pady=(0, 6))
    quiz_btn = ttk.Button(
        frame, text="STRUCTURE CARDS", command=lambda: on_start("structure"), style="Accent.TButton",
    )
    quiz_btn.pack(padx=12, pady=(0, 6), fill="x")
    # Shortened from "IDENTIFY IN SENTENCE"/"EXPORT GRAMMAR TO ANKI" —
    # confirmed directly (screenshot) that the longer text overflowed
    # past the button's own edge, since ttk.Button neither wraps nor
    # shrinks its label to fit.
    identify_btn = ttk.Button(
        frame, text="\U0001F50D IDENTIFY GRAMMAR", command=lambda: on_start("identify"),
    )
    identify_btn.pack(padx=12, pady=(0, 6), fill="x")
    anki_btn = ttk.Button(frame, text="EXPORT TO ANKI", command=on_export_anki)
    anki_btn.pack(padx=12, pady=(0, 12), fill="x")

    if themed is not None:
        themed.add(frame, bg="PANEL", highlightbackground="PANEL_BORDER")
        themed.add(header, bg="PANEL", fg="FG")
        themed.add(book_label, bg="PANEL", fg="FG")
        themed.add(meta_label, bg="PANEL", fg="DIM")
        themed.add(readiness_label, bg="PANEL", fg="ACCENT")
        themed.add(known_label, bg="PANEL", fg="FG")
        themed.add(to_learn_label, bg="PANEL", fg="FG")
        themed.add(count_label, bg="PANEL", fg="DIM")

    widgets = {
        "book_label": book_label, "meta_label": meta_label, "readiness_label": readiness_label,
        "progress_bar": progress_bar, "known_label": known_label, "to_learn_label": to_learn_label,
        "count_label": count_label, "quiz_btn": quiz_btn, "identify_btn": identify_btn, "anki_btn": anki_btn,
    }
    return frame, widgets
