"""Run evaluation experiments:  python -m evaluation <experiment|all> [options]"""
import argparse
import sys
import time
from pathlib import Path

from . import extraction, preprocessing, resolution, security, summarization, system

EXPERIMENTS = {m.NAME: m for m in (system, preprocessing, extraction, resolution, summarization, security)}
DEFAULT_SAMPLES = {"system": 5, "preprocessing": 3, "extraction": 3, "resolution": 3, "summarization": 3,
                   "security": None}   # security: None = every attack payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m evaluation", description="Benchmark the knowledge-graph pipeline (needs a running Ollama).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="experiments:\n" + "\n".join(f"  {n:<14}{m.DESCRIPTION}" for n, m in EXPERIMENTS.items()))
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("-n", "--samples", type=int, help="articles to evaluate (security: attacks per module)")
    parser.add_argument("--source", choices=["bbc", "local"], default="bbc",
                        help="bbc = RealTimeData/bbc_news_alltime (needs the dataset); local = data/*.txt")
    parser.add_argument("--month", default="2025-01", help="BBC dataset config, e.g. 2025-01")
    parser.add_argument("--offset", type=int, default=0, help="skip this many articles (use a different slice)")
    parser.add_argument("--max-chars", type=int, default=0, help="system: truncate articles to this many characters")
    parser.add_argument("--windows", type=int, default=6, help="extraction: sliding windows per article")
    parser.add_argument("--gold", help="resolution: hand-labelled pairs file (default: evaluation/gold/resolution_pairs.json)")
    parser.add_argument("--llm-judge", action="store_true",
                        help="resolution: let the LLM judge pairs that have no hand label (lenient, optimistic)")
    parser.add_argument("--out", help="directory for result files (default: evaluation/results)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from src.llm import health

    ok, status = health(timeout=3.0, lang="en")
    if not ok:
        sys.exit(f"Ollama is not ready: {status}")

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        run_args = argparse.Namespace(**vars(args))
        run_args.samples = args.samples if args.samples is not None else DEFAULT_SAMPLES[name]
        print(f"\n=== {name}: {EXPERIMENTS[name].DESCRIPTION} ===")
        started = time.time()
        report = EXPERIMENTS[name].run(run_args)
        report.notes.append(f"Wall time: {time.time() - started:.0f}s; model and settings from src/config.py.")
        print("\n" + report.markdown())
        kwargs = {"directory": Path(args.out)} if args.out else {}
        for path in report.save(**kwargs):
            print(f"saved {path}")


if __name__ == "__main__":
    main()
