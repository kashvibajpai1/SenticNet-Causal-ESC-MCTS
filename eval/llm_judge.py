"""Pairwise LLM-as-judge evaluation (Part C of causal_esc_mcts_fix_spec.md).

Addresses the "no human eval" review point. Separate from the local Qwen
pipeline entirely: this calls an external, stronger model via API
(Anthropic's Messages API), not a local model -- no weights/GPU concern.

Design (per spec):
- Pairwise comparison, not absolute 1-10 scoring: causal_mcts vs
  flow_ablation, and causal_mcts vs random_floor, over the same (larger)
  eval set produced by B2 -- not a separate smaller sample.
- Bias control: which candidate is shown as "A" vs "B" is randomized per
  call (seeded, so the run is reproducible) so position bias cancels out
  over the sample. The judge is never told which system produced which
  reply.
- Ties are allowed.
- Output: win/loss/tie counts per pairing (results/judge/judge_seed{N}.json,
  for the results table) plus a handful of full transcripts with the
  judge's reasoning (for the paper's qualitative section, alongside
  eval/qualitative.py's output).

Deferred execution: this environment has no ANTHROPIC_API_KEY set, so this
module is built and tested (pure logic: prompt construction, the bias-flip
winner-mapping, response parsing -- see tests/test_llm_judge.py, no network
calls) but was not run. Set ANTHROPIC_API_KEY and run:
    python -m eval.llm_judge --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Callable
from typing import Any

DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5")
PAIRINGS = [("causal_mcts", "flow_ablation"), ("causal_mcts", "random_floor")]

JUDGE_SYSTEM_PROMPT = (
    "You are judging two candidate replies from an AI emotional-support "
    "conversation agent. You will see the dialogue context and two "
    "candidate replies, labeled A and B. Decide which reply is the better "
    "supportive response -- more emotionally attuned, appropriate to the "
    "conversation phase, and helpful -- or say it is a tie. "
    'Respond with only a JSON object: {"winner": "A" | "B" | "tie", '
    '"reasoning": "<1-3 sentences>"}. You were not told which system '
    "produced which reply, and it is irrelevant to the judgment."
)


def build_user_prompt(context: list[str], reply_a: str, reply_b: str) -> str:
    """Build the pairwise-comparison prompt for one eval instance."""
    lines = []
    for i, turn in enumerate(context):
        speaker = "Seeker" if i % 2 == 0 else "Supporter"
        lines.append(f"{speaker}: {turn}")
    convo = "\n".join(lines)
    return (
        f"Conversation so far:\n{convo}\n\n"
        f"Candidate reply A: {reply_a}\n\n"
        f"Candidate reply B: {reply_b}\n\n"
        "Which reply is the better supportive response? Respond with only "
        "the JSON object described in your instructions."
    )


def parse_judge_response(text: str) -> dict[str, str]:
    """Parse the judge's JSON reply; falls back to a tie on malformed output."""
    try:
        parsed = json.loads(text)
        winner = parsed.get("winner", "tie")
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError, TypeError):
        winner, reasoning = "tie", f"[unparsed judge output] {text[:200]}"
    if winner not in ("A", "B", "tie"):
        winner = "tie"
    return {"winner": winner, "reasoning": reasoning}


def resolve_winner_system(
    judge_winner_label: str, system_a: str, system_b: str, flipped: bool
) -> str:
    """Map the judge's A/B choice back to the real system names.

    `flipped` records whether reply_a/reply_b (the real systems, in their
    canonical order) were swapped before being shown to the judge as A/B.
    """
    if judge_winner_label == "tie":
        return "tie"
    label_is_a = judge_winner_label == "A"
    # If we flipped when presenting, the judge's "A" is really system_b.
    picked_first = label_is_a != flipped
    return system_a if picked_first else system_b


CallJudgeFn = Callable[[list[str], str, str], dict[str, str]]


def run_pairing(
    call_judge: CallJudgeFn,
    system_a: str,
    system_b: str,
    instances: list[dict[str, Any]],
    preds_a: list[str],
    preds_b: list[str],
    *,
    seed: int,
    n_transcripts: int = 8,
) -> dict[str, Any]:
    """Run one pairwise comparison over all instances.

    `call_judge(context, shown_a, shown_b) -> {"winner": ..., "reasoning": ...}`
    is injected so the judge-calling logic can be tested without a live API.
    """
    if len(instances) != len(preds_a) or len(instances) != len(preds_b):
        raise ValueError("instances/preds_a/preds_b must be the same length")

    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    transcripts: list[dict[str, Any]] = []

    for inst, reply_a, reply_b in zip(instances, preds_a, preds_b):
        # Randomize which system is labeled "A" so position bias cancels
        # out over the sample; the judge is never told which is which.
        flipped = rng.random() < 0.5
        shown_a, shown_b = (reply_b, reply_a) if flipped else (reply_a, reply_b)

        result = call_judge(inst["context"], shown_a, shown_b)
        winner_system = resolve_winner_system(result["winner"], system_a, system_b, flipped)
        counts[winner_system] += 1

        if len(transcripts) < n_transcripts:
            transcripts.append(
                {
                    "conversation_id": inst.get("conversation_id"),
                    "context": inst["context"],
                    "reply_a_system": system_b if flipped else system_a,
                    "reply_b_system": system_a if flipped else system_b,
                    "reply_a": shown_a,
                    "reply_b": shown_b,
                    "judge_winner_label": result["winner"],
                    "judge_winner_system": winner_system,
                    "judge_reasoning": result["reasoning"],
                }
            )

    return {
        "pairing": f"{system_a}_vs_{system_b}",
        "n_instances": len(instances),
        "wins": {
            system_a: counts[system_a],
            system_b: counts[system_b],
            "tie": counts["tie"],
        },
        "transcripts": transcripts,
    }


def _make_anthropic_call_judge(client: Any, model: str) -> CallJudgeFn:
    def _call(context: list[str], reply_a: str, reply_b: str) -> dict[str, str]:
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_prompt(context, reply_a, reply_b)}
            ],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return parse_judge_response(text)

    return _call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=42, help="Eval seed whose results/metrics/*.json to compare."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-transcripts", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "[llm_judge] ANTHROPIC_API_KEY not set. This harness is built and "
            "tested (tests/test_llm_judge.py, no network calls) but execution "
            "is deferred until credentials are provided. Set ANTHROPIC_API_KEY "
            "and re-run: python -m eval.llm_judge --seed 42",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print(
            "[llm_judge] the `anthropic` package is not installed. "
            "Install it with `pip install anthropic` and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from eval.data import build_eval_instances

    metrics_dir = os.path.join(root, "results", "metrics")
    per_system: dict[str, dict[str, Any]] = {}
    needed_systems = {s for pair in PAIRINGS for s in pair}
    for sys_name in needed_systems:
        path = os.path.join(metrics_dir, f"{sys_name}_seed{args.seed}.json")
        if not os.path.isfile(path):
            print(
                f"[llm_judge] missing {path}; run `python -m eval.run_eval "
                f"--system {sys_name} --seed {args.seed} --n <N>` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(path) as f:
            per_system[sys_name] = json.load(f)

    n = per_system["causal_mcts"]["n_instances"]
    instances = build_eval_instances(root, n=n)

    client = anthropic.Anthropic(api_key=api_key)
    call_judge = _make_anthropic_call_judge(client, args.model)

    results = []
    for system_a, system_b in PAIRINGS:
        preds_a = per_system[system_a]["per_example"]["prediction"]
        preds_b = per_system[system_b]["per_example"]["prediction"]
        result = run_pairing(
            call_judge,
            system_a,
            system_b,
            instances,
            preds_a,
            preds_b,
            seed=args.seed,
            n_transcripts=args.n_transcripts,
        )
        results.append(result)
        print(f"[llm_judge] {result['pairing']}: {result['wins']}")

    out_dir = os.path.join(root, "results", "judge")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"judge_seed{args.seed}.json")
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "pairings": results}, f, indent=2)
    print(f"[llm_judge] wrote {summary_path}")


if __name__ == "__main__":
    main()
