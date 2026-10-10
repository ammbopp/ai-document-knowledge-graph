# relation_extraction_ai.py
import re
import threading
from functools import lru_cache

import spacy

from .llm import generate
from .ontology import ENTITY_TYPES

nlp = spacy.load("en_core_web_sm")


# ==========================================
# 1. Prompt Engineering
# ==========================================
PROMPT = """You are an expert NLP relation extraction system. Your task is to extract highly accurate relationship triplets (Subject | Verb Phrase | Object) from the text.

CRITICAL RULES:
1. Output MUST strictly follow this pattern: Subject | Verb Phrase | Object ###
2. Separate triplets using '###'. You can put multiple triplets on new lines.
3. Keep Entities SHORT and concrete (1-4 words).
4. DIRECTIONAL ACCURACY: Pay close attention to who performs the action and who receives it. Do not invert subjects and objects.
5. CONTENT OVER SPEECH WRAPPERS: Never extract the act of speaking or reporting itself (e.g., "said", "told", "testified", "claimed") as a relation. Extract the core factual claim inside what was said.

=== EXAMPLES OF STRUCTURAL EXTRACTION ===
Input: Alice told investigators last week that Bob secretly stole the company assets from the vault.
Output: Bob | stole | company assets ### company assets | stored in | vault ###

Input: Several employees filed a lawsuit against the firm, claiming they were mistreated by the supervisor.
Output: employees | filed lawsuit against | firm ### supervisor | mistreated | employees ###
=== END OF EXAMPLES ===

STRICT INSTRUCTION: Extract triplets ONLY from the input text provided below. Do not copy the examples.

Input: {}
Output:"""



# ==========================================
# 2. Helper Functions
# ==========================================
_PRONOUNS = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}
_STOPWORDS = nlp.Defaults.stop_words


_nlp_lock = threading.Lock()  # extract_relations may run on several worker threads


@lru_cache(maxsize=4096)
def _parse(text):
    with _nlp_lock:
        return nlp(text)


def ask_llm(text):
    """Return the model's triples as 'S | V | O ### S | V | O' (empty string on failure)."""
    try:
        result = generate(PROMPT.format(text), num_predict=300, stop=["Input:", "=== END"])
    except Exception as e:
        print(f"\n❌ Ollama Error: {e}")
        return ""

    # Strip list/markup noise but keep apostrophes and quotes inside names (O'Brien, Trump's)
    result = result.replace("[", "").replace("]", "").replace("`", "")

    # Keep only lines containing triples; each line may hold several '###'-separated facts
    facts = []
    for line in result.split("\n"):
        if "|" in line:
            facts.extend(f.strip() for f in line.split("###") if f.strip())
    return " ### ".join(facts)


# ==========================================
# 3. Dynamic Structural Filters
# ==========================================
def is_valid_entity(entity):
    ent_clean = entity.strip().strip("\"'")
    is_year = ent_clean.isdigit() and 1800 <= int(ent_clean) <= 2100
    if not ent_clean or (ent_clean.isdigit() and not is_year) or len(ent_clean) < 2:
        return False

    if is_year:
        return True

    if ent_clean.lower() in _PRONOUNS:
        return False

    doc = _parse(ent_clean)

    # Rule 1: must contain a noun / proper noun / pronoun. spaCy tags unknown capitalised
    # names ("Anthropic") as ADJ/X in isolation, so those count as proper nouns.
    def _is_nominal(t):
        return t.pos_ in ("NOUN", "PROPN", "PRON") or (t.pos_ in ("ADJ", "X") and t.text[:1].isupper())

    if not any(_is_nominal(t) for t in doc):
        return False

    # Rule 2: no clauses inside a node. spaCy often tags names as VERB when parsing
    # them in isolation, so only reject AUX or verbs inside a lower-case phrase.
    if any(t.pos_ == "AUX" for t in doc):
        return False
    if any(t.pos_ == "VERB" and not t.text[:1].isupper() for t in doc):
        return False

    # Rule 3: must not end with a preposition/conjunction (dangling node)
    if doc[-1].pos_ in ("ADP", "CCONJ", "SCONJ", "PART") and not doc[-1].text[:1].isupper():
        return False

    return True


_SPEECH_VERBS = {
    "say", "says", "said", "tell", "tells", "told", "claim", "claims", "claimed", "state", "states",
    "stated", "testify", "testified", "add", "adds", "added", "note", "notes", "noted", "explain",
    "explains", "explained", "argue", "argues", "argued", "warn", "warns", "warned", "report",
    "reports", "reported", "reply", "replied", "insist", "insists", "insisted", "suggest",
    "suggests", "suggested", "assert", "asserted", "ask", "asks", "asked",
}


def is_valid_relation(relation):
    words = relation.strip().lower().split()
    if len(relation.strip()) < 2 or len(words) > 5:
        return False
    # Speech wrappers ("X said ...") are not facts
    if words[0] in _SPEECH_VERBS:
        return False
    return True


def is_hallucination(head, tail, original_text, min_overlap=0.5):
    """True if an entity's content words mostly do not appear (as whole words) in the source."""
    source_words = set(re.findall(r"[\w'-]+", original_text.lower()))

    def grounded(entity):
        words = [w for w in re.findall(r"[\w'-]+", entity.lower()) if w not in _STOPWORDS]
        if not words:
            return False
        return sum(w in source_words for w in words) / len(words) >= min_overlap

    return not (grounded(head) and grounded(tail))


def parse_triples(text):
    triples = []
    facts = [f.strip() for f in re.split(r"###|\n", text) if f.strip()]
    for fact in facts:
        parts = [p.strip() for p in fact.split("|") if p.strip()]
        if len(parts) == 3:
            triples.append((parts[0], parts[1], parts[2]))
    return triples


@lru_cache(maxsize=2048)
def relation_key(relation):
    """Tense/aux-insensitive key: 'will be causing' and 'cause' compare equal."""
    lemmas = [t.lemma_.lower() for t in _parse(relation.strip()) if t.pos_ != "AUX" and not t.is_punct]
    return " ".join(lemmas) or relation.strip().lower()


def deduplicate(relations):
    seen = set()
    unique = []
    for r in relations:
        key = (r["head"].lower(), relation_key(r["relation"]), r["tail"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def infer_entity_type(entity, doc):
    """Map an entity to an ontology type using spaCy NER over its source text."""
    target = entity.lower().strip()
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        text = ent.text.lower().strip()
        if text == target or target in text:
            return ENTITY_TYPES[ent.label_]
        # Entity = NER span + at most one extra word ("President Donald Trump"), not a longer phrase
        if text in target and len(target.split()) - len(text.split()) <= 1:
            return ENTITY_TYPES[ent.label_]
    return "Entity"


# ==========================================
# 4. Directional & Substance Gatekeeper
# ==========================================
_DIRECTION_PROMPT = """Text: "{sentence}"
Relation phrase: "{relation}"
Entities: A = {a}; B = {b}
In this text, which entity is the SUBJECT of the relation phrase (the one that does it or is described by it) and which is the OBJECT? Answer with ONLY the letter of the SUBJECT: A or B.
Answer:"""


def _pick_subject(sentence, relation, a, b):
    out = generate(_DIRECTION_PROMPT.format(sentence=sentence, relation=relation, a=a, b=b), num_predict=3)
    match = re.search(r"\b([AB])\b", out.upper())
    return match.group(1) if match else None


def check_direction(head, relation, tail, sentence):
    """
    False only when the LLM consistently names `tail` as the subject in both presentation orders.
    Mixed answers (A,A / B,B) just mean position bias, so they do not reject the triple.
    """
    first = _pick_subject(sentence, relation, head, tail)    # "B" -> tail is subject
    second = _pick_subject(sentence, relation, tail, head)   # "A" -> tail is subject
    return not (first == "B" and second == "A")


def evaluate_triplet_cloze(head, relation, tail, source_sentence, threshold=0.6):
    """
    LLM gate for direction and substance. Returns (is_valid, confidence in [0, 1]).
    The model rates the triplet 0-10; confidence = rating / 10 and the triplet passes when
    confidence >= threshold. A failed call rejects the triplet (fail closed).
    """
    prompt = f"""You are a strict validation logic gate for Knowledge Graphs. Rate the extracted triplet against the text.

Text: "{source_sentence}"
Triplet: [{head}] ---> ({relation}) ---> [{tail}]

CRITICAL VALIDATION RULES:
1. DIRECTIONAL CHECK: Does [{head}] actually initiate or perform the action ({relation}) upon [{tail}] in the text? If the text states or implies the reverse, it is an INVERSION and must score 0.
2. SUBSTANCE CHECK: Is the relation a real factual event, or just a speech tag like "told", "said", "testified"? Speech tags must score 0.
3. FAITHFULNESS: Is the triplet explicitly supported by the text? Unsupported or invented facts must score 0.

Reply with ONLY an integer from 0 (invalid) to 10 (perfectly valid).
Score:"""

    try:
        result = generate(prompt, num_predict=8)
        # The 0-10 score cannot tell direction apart (both orders score alike), so ask a
        # forced-choice question in both orders; the head must be picked as subject both times.
        if not check_direction(head, relation, tail, source_sentence):
            print(f"↔️  [Direction rejected] {head} -> {relation} -> {tail}")
            return False, 0.0
    except Exception as e:
        print(f"⚠️ Evaluator Error: {e}")
        return False, 0.0

    match = re.search(r"\d+", result)
    if match:
        confidence = min(int(match.group()), 10) / 10
    else:
        confidence = 1.0 if result.upper().startswith("VALID") else 0.0

    if confidence < threshold:
        print(f"🚫 [Structural Filtered: REJECTED] {head} -> {relation} -> {tail}")
        return False, confidence
    return True, confidence


# ==========================================
# 5. Main Extraction Function
# ==========================================
_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_POSSESSOR_PREFIX = re.compile(r"^.+?(?:['’]s|(?<=s)['’])\s+(?=\S)")
_POSSESSIVE_SUFFIX = re.compile(r"(?:['’]s|(?<=s)['’])$")

_PASSIVE_MARKERS = {"is", "are", "was", "were", "be", "been", "being", "get", "gets", "got", "gotten"}
_PASSIVE_AUX = _PASSIVE_MARKERS | {"has", "have", "had", "will", "would", "can", "could", "may", "might", "must", "should"}
# Irregular past participles that are not past-tense forms (so "stood by" stays active)
_IRREGULAR_PARTICIPLES = {
    "built", "written", "led", "held", "made", "taken", "given", "sold", "bought", "won", "known",
    "found", "chosen", "driven", "shown", "seen", "done", "begun", "broken", "caught", "paid",
    "set", "put", "run", "struck", "hit", "cut", "sent", "lost", "kept", "left", "brought",
}
_TIME_PREPS = r"on|in|by|at|since|until|from|during|before|after"


def clean_entity_name(name):
    """Strip articles and possessives: "Trump's previous taxes" -> "previous taxes", "Trump's" -> "Trump"."""
    name = _ARTICLE.sub("", name.strip())
    name = _POSSESSOR_PREFIX.sub("", name)
    name = _POSSESSIVE_SUFFIX.sub("", name).strip()
    return _ARTICLE.sub("", name)


def normalize_passive(head, relation, tail):
    """
    Turn a passive triple into its active form by swapping the arguments:
    (White House, contacted by, BBC) -> (BBC, contacted, White House).
    """
    words = relation.strip().split()
    if len(words) < 2 or words[-1].lower() != "by":
        return head, relation, tail

    body = words[:-1]
    had_aux = False
    while body and body[0].lower() in _PASSIVE_AUX:
        had_aux = had_aux or body[0].lower() in _PASSIVE_MARKERS
        body = body[1:]
    if not body:
        return head, relation, tail

    verb = body[0].lower()
    is_participle = verb in _IRREGULAR_PARTICIPLES or (verb.endswith("ed") and len(verb) > 3)
    if not (had_aux or is_participle):
        return head, relation, tail
    return tail, " ".join(body), head


def is_bare_passive(relation):
    """'has been suspended' (passive, no 'by' agent, no preposition): the direction is unreliable."""
    words = relation.strip().lower().split()
    if len(words) < 2 or words[-1] == "by":
        return False
    has_marker = any(w in _PASSIVE_MARKERS for w in words[:-1])
    last = words[-1]
    return has_marker and (last.endswith("ed") or last in _IRREGULAR_PARTICIPLES)


def split_trailing_date(entity):
    """
    If the entity ends with a DATE (per spaCy NER) and something else precedes it, return
    (rest, preposition, date_text); otherwise None. "force on Tuesday, 24 February" ->
    ("force", "on", "Tuesday, 24 February").
    """
    stripped = entity.rstrip(" .,")
    dates = [e for e in _parse(stripped).ents if e.label_ == "DATE"]
    if not dates or dates[-1].end_char < len(stripped):
        return None
    # Merge adjacent DATE spans ("Tuesday" + ", " + "24 February") into one date expression
    start = dates[-1].start_char
    for prev in reversed(dates[:-1]):
        if not stripped[prev.end_char:start].strip(" ,"):
            start = prev.start_char
        else:
            break
    date_text = stripped[start:]
    rest = stripped[:start].strip(" ,")
    prep_match = re.search(rf"\b({_TIME_PREPS})$", rest, re.IGNORECASE)
    prep = "on"
    if prep_match:
        prep = prep_match.group(1).lower()
        rest = rest[:prep_match.start()].strip(" ,")
    if not rest or not is_valid_entity(rest):
        return None
    return rest, prep, date_text


def find_grounding_sentence(source_doc, head, tail):
    """First sentence of the chunk that contains both entities, or None (cross-sentence triples are dropped)."""
    for sent in source_doc.sents:
        if not is_hallucination(head, tail, sent.text):
            return sent.text.strip()
    return None


def extract_relations(sentences):
    print("🔥 extract_relations CALLED (Ollama)")
    relations = []

    for sent in sentences:
        if len(sent) < 10:
            continue

        try:
            output = ask_llm(sent)
            if not output:
                continue

            print("DEBUG Output:")
            for fact in output.split("###"):
                if fact.strip():
                    print(f"  {fact.strip()}")

            source_doc = _parse(sent)

            for head, rel_text, tail in parse_triples(output):
                head, rel_text, tail = normalize_passive(
                    clean_entity_name(head), rel_text.strip(), clean_entity_name(tail))

                if not is_valid_entity(head) or not is_valid_entity(tail):
                    continue
                if not is_valid_relation(rel_text) or is_bare_passive(rel_text):
                    continue
                if head.lower() == tail.lower():
                    continue
                # Both entities must appear in ONE sentence; verify and cite against that sentence only
                evidence = find_grounding_sentence(source_doc, head, tail)
                if evidence is None:
                    continue

                is_valid, confidence_score = evaluate_triplet_cloze(head, rel_text, tail, evidence)
                if not is_valid:
                    continue

                record = {
                    "source_sentence": evidence,
                    "confidence": confidence_score,
                    "method": "Ollama_Factual_Verifier",
                }

                # Separate a trailing date into its own Time node: (force, on, Tuesday)
                date_part = split_trailing_date(tail)
                if date_part:
                    tail, prep, date_text = date_part
                    relations.append({
                        **record, "head": tail, "relation": f"occurs {prep}", "tail": date_text,
                        "head_type": infer_entity_type(tail, source_doc), "tail_type": "Time",
                    })

                relations.append({
                    **record, "head": head, "relation": rel_text, "tail": tail,
                    "head_type": infer_entity_type(head, source_doc),
                    "tail_type": infer_entity_type(tail, source_doc),
                })

        except Exception as e:
            print(f"❌ Error processing sentence: {e}")

    final_relations = deduplicate(relations)
    print(f"✅ Extracted {len(final_relations)} valid relations.")
    return final_relations
