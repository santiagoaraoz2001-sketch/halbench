"""Command-line interface: `halbench {run,score,verify,submit}`."""
from __future__ import annotations
import argparse
import json
import os
import sys

from halbench import BENCHMARK_VERSION


def cmd_run(args: argparse.Namespace) -> int:
    from halbench.runner import run_model
    summary = run_model(
        backend_name=args.backend,
        model=args.model,
        out_dir=args.out,
        corpus_source=args.corpus,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system=args.system,
        concurrency=args.concurrency,
        limit=args.limit,
        resume=not args.no_resume,
        verbose=not args.quiet,
        score=not args.no_score,
    )
    print(f"\n{json.dumps(summary, indent=2)}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from halbench.scoring import score_responses
    out_path = args.out or args.responses.replace("responses", "scores")
    if out_path == args.responses:
        out_path = args.responses + ".scores.jsonl"
    score_responses(args.responses, out_jsonl=out_path,
                     max_workers=args.concurrency, verbose=not args.quiet)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from halbench.verify import verify_submission
    result = verify_submission(
        args.submission,
        expected_n=args.expected_n,
        tolerance=args.tolerance,
        verbose=not args.quiet,
    )
    print("\n=== Verification result ===")
    print(json.dumps(result["summary"], indent=2))
    if result["ok"]:
        print("\nOK — submission verifies cleanly.")
        return 0
    print("\nFAILED:")
    for err in result["errors"]:
        print(f"  - {err}")
    return 1


def cmd_submit(args: argparse.Namespace) -> int:
    """Print PR instructions. We deliberately don't auto-open a PR — submitters
    should review their scores before opening one, and we don't want to require
    a GitHub token from people who just want to evaluate locally."""
    import shutil
    fn = os.path.basename(args.submission)
    model_slug = args.model_id.replace("/", "__") if args.model_id else fn.split(".")[0]
    dest = f"leaderboard/submissions/{model_slug}.jsonl"

    print(f"""
To submit {fn} to the public leaderboard:

  1. Fork  https://github.com/santiagoaraoz/halbench  on GitHub.
  2. Clone your fork and copy the submission in:

       mkdir -p leaderboard/submissions
       cp {args.submission} {dest}

  3. (Optional) Add a SUBMITTER.md under leaderboard/submissions/{model_slug}/
     with your name/email and any notes about sampling params used.

  4. Open a PR. The verify-submission GitHub Action will:
       - Re-score every response against the locked calibration endpoints
       - Confirm scores match within ±0.005
       - Block merge if any tampering is detected

  5. On merge, the Space rebuilds automatically with your model included.

You can pre-verify locally before opening the PR:

       halbench verify {args.submission}
""")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="halbench",
        description=f"HalBench {BENCHMARK_VERSION} — sycophancy benchmark for frontier LLMs",
    )
    p.add_argument("--version", action="version", version=f"halbench {BENCHMARK_VERSION}")
    subs = p.add_subparsers(dest="cmd", required=True)

    # ---- run ----
    r = subs.add_parser("run", help="Run a model end-to-end (chat + score)")
    r.add_argument("--model", required=True, help="Model id, e.g. anthropic/claude-sonnet-4.6")
    r.add_argument("--backend", default="openrouter", help="Backend (default: openrouter)")
    r.add_argument("--out", default="./halbench_run", help="Output directory")
    r.add_argument("--corpus", default=None, help="Local corpus path (else: HF Dataset)")
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--system", default=None, help="Optional system prompt")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--limit", type=int, default=None, help="Run only first N items (smoke test)")
    r.add_argument("--no-resume", action="store_true", help="Don't skip already-run items")
    r.add_argument("--no-score", action="store_true", help="Just produce responses.jsonl; skip scoring")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    # ---- score ----
    s = subs.add_parser("score", help="Score a responses.jsonl produced by your own runner")
    s.add_argument("responses", help="Path to responses.jsonl")
    s.add_argument("--out", default=None, help="Output scores.jsonl (default: alongside input)")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_score)

    # ---- verify ----
    v = subs.add_parser("verify", help="Verify a scores.jsonl can be reproduced from its responses")
    v.add_argument("submission", help="Path to scores.jsonl")
    v.add_argument("--expected-n", type=int, default=3600)
    v.add_argument("--tolerance", type=float, default=0.005)
    v.add_argument("--quiet", action="store_true")
    v.set_defaults(func=cmd_verify)

    # ---- submit ----
    sb = subs.add_parser("submit", help="Print instructions for opening a leaderboard PR")
    sb.add_argument("submission", help="Path to scores.jsonl")
    sb.add_argument("--model-id", default=None, help="Override model slug (default: from filename)")
    sb.set_defaults(func=cmd_submit)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
