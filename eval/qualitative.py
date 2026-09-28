"""Extract side-by-side qualitative examples from the causal_mcts seed=42 eval run.

Writes results/qualitative/examples.jsonl and a human-readable
results/qualitative/examples.md. Addresses the reviewers' complaint that
qualitative analysis was claimed as "omitted for space" despite the paper
being under the page limit -- this repo previously had none at all.
"""

from __future__ import annotations

import json
import os

from eval.data import build_eval_instances

N_EXAMPLES = 18
SEED = 42


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(root, "results", "metrics", f"causal_mcts_seed{SEED}.json")
    with open(metrics_path) as f:
        eval_result = json.load(f)

    per_example = eval_result["per_example"]
    n = len(per_example["prediction"])
    instances = build_eval_instances(root, n=n)

    examples = []
    for i in range(min(N_EXAMPLES, n)):
        examples.append(
            {
                "conversation_id": instances[i]["conversation_id"],
                "context": instances[i]["context"],
                "predicted_strategy": per_example["predicted_strategy"][i],
                "gold_strategy": per_example["gold_strategy"][i],
                "strategy_match": per_example["predicted_strategy"][i]
                == per_example["gold_strategy"][i],
                "generated_reply": per_example["prediction"][i],
                "gold_reply": per_example["reference"][i],
                "rougeL": per_example["rougeL"][i],
                "bertscore_f1": per_example["bertscore_f1"][i],
            }
        )

    out_dir = os.path.join(root, "results", "qualitative")
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "examples.jsonl")
    with open(jsonl_path, "w") as f:
        f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in examples)

    md_lines = ["# Qualitative examples (causal_mcts, seed=42)\n"]
    for i, ex in enumerate(examples):
        md_lines.append(f"## Example {i + 1} ({ex['conversation_id']})\n")
        md_lines.append("**Context:**")
        for j, turn in enumerate(ex["context"][-4:]):
            speaker = "Seeker" if j % 2 == 0 else "Supporter"
            md_lines.append(f"- {speaker}: {turn}")
        match_str = "match" if ex["strategy_match"] else "mismatch"
        md_lines.append(
            f"\n**Strategy:** predicted=`{ex['predicted_strategy']}` "
            f"gold=`{ex['gold_strategy']}` ({match_str})"
        )
        md_lines.append(f"\n**Generated reply:** {ex['generated_reply']}")
        md_lines.append(f"\n**Gold reply:** {ex['gold_reply']}")
        md_lines.append(f"\n**ROUGE-L:** {ex['rougeL']:.3f} | **BERTScore F1:** {ex['bertscore_f1']:.3f}\n")

    md_path = os.path.join(out_dir, "examples.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"[qualitative] wrote {jsonl_path} and {md_path} ({len(examples)} examples)")


if __name__ == "__main__":
    main()
