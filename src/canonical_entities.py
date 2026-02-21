# canonical_entities.py
import re

class EntityCanonicalizer:
    def __init__(self):
        self.entity_map = {}

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def canonicalize(self, entity: str) -> str:
        key = self.normalize(entity)

        if key in self.entity_map:
            return self.entity_map[key]

        canonical = entity.strip().title()
        self.entity_map[key] = canonical
        return canonical
