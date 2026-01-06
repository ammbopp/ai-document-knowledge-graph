import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str):
    """
    Extract named entities from text
    """
    doc = nlp(text)
    entities = set()

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "WORK_OF_ART", "DATE"]:
            entities.add(ent.text)

    return list(entities)
