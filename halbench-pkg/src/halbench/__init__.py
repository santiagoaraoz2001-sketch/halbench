"""HalBench V2.1 — a sycophancy/hallucination benchmark for frontier LLMs.

Quickstart:

    >>> from halbench import load_corpus, score_responses
    >>> items = load_corpus()
    >>> # Run your model on each item["prompt"], save responses as JSONL with
    >>> # one row per item: {"item_id": ..., "response_text": ...}
    >>> scores = score_responses("my_model_responses.jsonl")

CLI:

    $ halbench run --model anthropic/claude-sonnet-4.6 --backend openrouter
    $ halbench score my_model_responses.jsonl
    $ halbench verify scores.jsonl
    $ halbench submit scores.jsonl

See README.md for the full submission workflow.
"""
from halbench.corpus import load_corpus, load_endpoints
from halbench.scoring import score_response, score_responses, GENERIC_ANCHORS_M5
from halbench.embedder import HarrierEmbedder

__version__ = "2.2.0"
BENCHMARK_VERSION = "v2.2.0"

__all__ = [
    "load_corpus",
    "load_endpoints",
    "score_response",
    "score_responses",
    "HarrierEmbedder",
    "GENERIC_ANCHORS_M5",
    "BENCHMARK_VERSION",
]
