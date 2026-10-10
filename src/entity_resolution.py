from collections import Counter
from functools import lru_cache

import numpy as np
import spacy

from .config import RESOLVER_THRESHOLD
from .relation_extraction_ai import relation_key

nlp = spacy.load("en_core_web_sm")

_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedder():
    """Load the sentence-embedding model (expensive: cache it, but never share a resolver)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


@lru_cache(maxsize=4096)
def simplify_entity_name(text):
    """
    Drop modifiers (adjectives, adverbs, prepositions...) and keep nouns, proper nouns and
    numbers, so 'Windows 11' and 'Apple Watch' stay distinct from 'Windows' and 'Apple'.
    """
    doc = nlp(text)
    clean_tokens = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN", "NUM")]
    return " ".join(clean_tokens) if clean_tokens else text


def _token_subset(a, b):
    """True if the shorter token list is contained in the longer and differs by at most 1 token."""
    short, long_ = sorted((a.split(), b.split()), key=len)
    return bool(short) and set(short) <= set(long_) and len(long_) - len(short) <= 1


def _surname_match(a, b):
    """'Bekker' <-> 'Koos Bekker': a single capitalised word equal to the last word of a multi-word name."""
    short, long_ = sorted((a.split(), b.split()), key=len)
    return (len(short) == 1 and len(long_) >= 2 and short[0][:1].isupper() and len(short[0]) > 3
            and all(w[:1].isupper() for w in long_) and short[0].lower() == long_[-1].lower())


class EntityResolver:
    """
    Merges surface forms of the same entity into clusters. One instance = one document:
    create a new resolver per analysis so entities never leak between documents.
    """

    def __init__(self, threshold=RESOLVER_THRESHOLD, model=None):
        self.model = model if model is not None else load_embedder()
        self.threshold = threshold
        self.key_to_cluster = {}   # simplified lower-case key -> cluster id
        self.cluster_name = {}     # cluster id -> current canonical display name
        self.cluster_type = {}     # cluster id -> entity type
        self.cluster_forms = {}    # cluster id -> Counter of surface forms seen
        self._keys = []            # embedded keys (parallel to _embeddings rows)
        self._embeddings = None    # L2-normalised matrix

    # ---- helpers ----
    @staticmethod
    def _types_match(existing_type, new_type):
        if existing_type == "Entity" or new_type == "Entity":
            return True
        return existing_type == new_type

    def _embed(self, key):
        vec = np.asarray(self.model.encode([key]), dtype=float)[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def _register(self, key, cid):
        self.key_to_cluster[key] = cid
        vec = self._embed(key)
        self._embeddings = vec[None, :] if self._embeddings is None else np.vstack([self._embeddings, vec])
        self._keys.append(key)

    def _join(self, cid, entity, entity_type):
        """Update a cluster with a newly seen surface form; the longest name becomes canonical."""
        entity = entity.strip()
        forms = self.cluster_forms[cid]
        forms[entity] += 1
        # Most frequent surface form wins (ties: longest) so a rare long phrase never renames a node
        self.cluster_name[cid] = max(forms, key=lambda f: (forms[f], len(f)))
        if self.cluster_type[cid] == "Entity" and entity_type != "Entity":
            self.cluster_type[cid] = entity_type

    # ---- public API ----
    def resolve(self, entity, entity_type="Entity"):
        entity = entity.strip()
        key = simplify_entity_name(entity).lower().strip()

        # 1. Exact key seen before
        cid = self.key_to_cluster.get(key)
        if cid is not None and self._types_match(self.cluster_type[cid], entity_type):
            self._join(cid, entity, entity_type)
            return self.cluster_name[cid]

        # 2. Word-level containment ("Elon Musk" ~ "Musk")
        for other, other_cid in self.key_to_cluster.items():
            # A bare surname matches its full name even if NER typed the two forms differently
            if _surname_match(entity, self.cluster_name[other_cid]):
                self.key_to_cluster[key] = other_cid
                self._join(other_cid, entity, self.cluster_type[other_cid])
                return self.cluster_name[other_cid]
            if not self._types_match(self.cluster_type[other_cid], entity_type):
                continue
            # Containment merging is only safe for typed named entities, not generic noun phrases
            specific = entity_type != "Entity" and self.cluster_type[other_cid] == entity_type
            if specific and len(key) > 3 and len(other) > 3 and _token_subset(key, other):
                self.key_to_cluster[key] = other_cid
                self._join(other_cid, entity, entity_type)
                return self.cluster_name[other_cid]

        # 3. Embedding similarity against compatible known keys
        if self._embeddings is not None:
            sims = self._embeddings @ self._embed(key)
            for idx in np.argsort(-sims):
                if sims[idx] < self.threshold:
                    break
                other_cid = self.key_to_cluster[self._keys[idx]]
                if self._types_match(self.cluster_type[other_cid], entity_type):
                    self._register(key, other_cid)
                    self._join(other_cid, entity, entity_type)
                    return self.cluster_name[other_cid]

        # 4. New cluster
        cid = len(self.cluster_name)
        self.cluster_name[cid] = entity
        self.cluster_type[cid] = entity_type
        self.cluster_forms[cid] = Counter({entity: 1})
        self._register(key, cid)
        return entity

    def canonical(self, entity):
        """Current canonical name for an already-resolved entity (reflects later upgrades)."""
        key = simplify_entity_name(entity.strip()).lower().strip()
        cid = self.key_to_cluster.get(key)
        return self.cluster_name[cid] if cid is not None else entity.strip()

    def type_of(self, entity, default="Entity"):
        key = simplify_entity_name(entity.strip()).lower().strip()
        cid = self.key_to_cluster.get(key)
        return self.cluster_type[cid] if cid is not None else default


# Generic role words that usually refer back to the one named person in an article
ROLE_WORDS = {"president", "prime minister", "minister", "the president", "ceo", "chairman",
              "chancellor", "governor", "mayor", "senator", "king", "queen"}


def _link_roles(resolver):
    """Point bare role nouns ("President") at the document's only Person cluster, if unambiguous."""
    people = [c for c, t in resolver.cluster_type.items() if t == "Person"]
    if len(people) != 1:
        return
    person = people[0]
    for key, cid in list(resolver.key_to_cluster.items()):
        if key in ROLE_WORDS and cid != person and resolver.cluster_type[cid] in ("Entity", "Person"):
            resolver.key_to_cluster[key] = person
            resolver.cluster_forms[person].update(resolver.cluster_forms[cid])
            resolver.cluster_name[person] = max(
                resolver.cluster_forms[person], key=lambda f: (resolver.cluster_forms[person][f], len(f)))


def resolve_relations(relations, resolver, on_progress=None):
    """
    Resolve head/tail of every relation, then relabel in a second pass so that earlier edges pick
    up canonical names that were upgraded later. Drops self-loops and exact duplicate triples;
    distinct relations (or directions) between the same pair of entities are kept.
    """
    staged = []
    for i, r in enumerate(relations):
        head = resolver.resolve(r["head"], r.get("head_type", "Entity"))
        tail = resolver.resolve(r["tail"], r.get("tail_type", "Entity"))
        staged.append((r, head, tail))
        if on_progress:
            on_progress(i + 1, len(relations))

    _link_roles(resolver)

    resolved, seen = [], set()
    for r, head, tail in staged:
        head, tail = resolver.canonical(head), resolver.canonical(tail)
        if head.lower() == tail.lower():
            continue
        key = (head.lower(), relation_key(r["relation"]), tail.lower())
        if key in seen:
            continue
        seen.add(key)
        new_r = r.copy()
        new_r.update(
            head=head,
            tail=tail,
            head_type=resolver.type_of(head, r.get("head_type", "Entity")),
            tail_type=resolver.type_of(tail, r.get("tail_type", "Entity")),
        )
        resolved.append(new_r)
    return resolved
