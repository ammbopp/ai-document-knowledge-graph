"""Shared helpers for the evaluation experiments: data loading, LLM judging, statistics, reporting."""
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET = "RealTimeData/bbc_news_alltime"
MIN_WORDS, MAX_WORDS = 80, 900  # skip stubs and very long pieces so runs stay comparable


# ---------- shared models (loaded once per process) ----------
_models = {}


def get_nlp():
    if "nlp" not in _models:
        import spacy

        _models["nlp"] = spacy.load("en_core_web_sm")
    return _models["nlp"]


def get_embedder():
    if "embedder" not in _models:
        from src.entity_resolution import load_embedder

        _models["embedder"] = load_embedder()
    return _models["embedder"]


# ---------- data ----------
@dataclass
class Article:
    id: str
    text: str
    title: str = ""
    reference: str = ""  # human-written summary (BBC "description"); empty for local files

    @property
    def words(self):
        return len(self.text.split())


def load_articles(n, *, source="bbc", month="2025-01", offset=0):
    """
    First `n` usable articles (deterministic order, so runs are comparable).
    source="bbc" streams the BBC dataset; source="local" uses data/*.txt (no reference summary).
    """
    if source == "local":
        files = sorted(LOCAL_DATA_DIR.glob("*.txt"))[offset:offset + n]
        return [Article(id=f.stem, text=f.read_text(encoding="utf-8").strip()) for f in files]

    from datasets import load_dataset

    data = load_dataset(DATASET, month, split="train")
    articles = []
    for i, row in enumerate(data):
        if i < offset:
            continue
        text = (row.get("content") or "").strip()
        if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
            continue
        reference = (row.get("description") or row.get("title") or "").strip()
        articles.append(Article(id=f"{month}#{i}", text=text, title=row.get("title") or "", reference=reference))
        if len(articles) == n:
            break
    return articles


# ---------- LLM judging ----------
def judge_json(prompt, retries=1):
    """
    Ask the model for a JSON verdict. Returns the parsed dict, or None if the call or parsing
    failed -- callers must exclude failures instead of inventing a default score.
    """
    from src.llm import generate

    for _ in range(retries + 1):
        try:
            data = json.loads(generate(prompt, temperature=0.0, json_mode=True, num_predict=200))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def as_bool(value):
    """Judge answers arrive as true/'yes'/1 in practice; anything unclear is None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "y", "1"):
            return True
        if v in ("false", "no", "n", "0"):
            return False
    return None


# ---------- statistics ----------
def mean_std(values):
    """(mean, sample std, n) over the non-null values; (nan, nan, 0) when empty."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan, math.nan, 0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt_stat(values, digits=2, pct=False):
    mean, std, n = mean_std(values)
    if n == 0:
        return "n/a"
    scale = 100 if pct else 1
    suffix = "%" if pct else ""
    return f"{mean * scale:.{digits}f}{suffix} ± {std * scale:.{digits}f}{suffix}"


def ratio(num, den):
    return num / den if den else math.nan


def prf(tp, fp, fn):
    """Precision / recall / F1 from counts; undefined (nan) when there is nothing to measure."""
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return precision, recall, f1


def pct(value, digits=1):
    return "n/a" if value is None or math.isnan(value) else f"{value * 100:.{digits}f}%"


# ---------- reporting ----------
@dataclass
class Report:
    """One experiment's output: per-item rows (CSV) plus a Markdown summary."""
    name: str
    title: str
    rows: list = field(default_factory=list)
    sections: list = field(default_factory=list)  # (heading, markdown table / text)
    notes: list = field(default_factory=list)

    def add_section(self, heading, body):
        self.sections.append((heading, body))

    def markdown(self):
        parts = [f"# {self.title}", ""]
        for heading, body in self.sections:
            parts += [f"## {heading}", "", body, ""]
        if self.notes:
            parts += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(parts)

    def save(self, directory=RESULTS_DIR):
        import pandas as pd

        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        md_path = directory / f"{self.name}_{stamp}.md"
        md_path.write_text(self.markdown(), encoding="utf-8")
        paths = [md_path]
        if self.rows:
            csv_path = directory / f"{self.name}_{stamp}.csv"
            pd.DataFrame(self.rows).to_csv(csv_path, index=False)
            paths.append(csv_path)
        return paths


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def words(text):
    return re.findall(r"[A-Za-z0-9'’-]+", text)
