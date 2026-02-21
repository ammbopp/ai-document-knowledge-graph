# ontology.py
ENTITY_TYPES = {
    "ORG": "Organization",
    "PERSON": "Person",
    "GPE": "Location",
    "DATE": "Time"
}

RELATION_SCHEMA = {
    ("Person", "Organization"): ["works_at", "studies_at"],
    ("Organization", "Organization"): ["collaborates_with", "partners_with"]
}
