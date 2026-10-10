import spacy
import re

from .ontology import ENTITY_TYPES

nlp = spacy.load("en_core_web_sm")

ENTITY_TYPE_MAP = ENTITY_TYPES

def normalize_entity(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def estimate_entity_confidence(ent):
    score = 1.0

    # สั้นเกินไป → noise
    if len(ent.text) < 2:
        score -= 0.4

    # ยาวเกินไป → chunk ผิด
    if len(ent.text.split()) > 6:
        score -= 0.3

    # วันที่ทั่วไป เช่น "2020"
    if ent.label_ == "DATE" and ent.text.isdigit():
        score -= 0.2

    return max(score, 0.0)

def extract_entities(text: str):
    """
    Thesis-level entity extraction with metadata
    """
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPE_MAP:
            continue

        normalized = normalize_entity(ent.text)
        confidence = estimate_entity_confidence(ent)

        if confidence < 0.5:
            continue

        key = (normalized.lower(), ENTITY_TYPE_MAP[ent.label_])

        # deduplicate แต่ยังเก็บ metadata
        if key not in entities:
            entities[key] = {
                "text": normalized,
                "type": ENTITY_TYPE_MAP[ent.label_],
                "source_sentence": text,
                "confidence": confidence,
                "method": "spaCy NER"
            }

    return list(entities.values())
