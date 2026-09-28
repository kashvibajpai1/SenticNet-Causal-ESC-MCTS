# Experiment Summary

Backbone: Qwen2.5-0.5B-Instruct (real, local).
Setup: causal_mcts, flow_ablation, random_floor. 3 seeds, 150 eval instances per seed.

Metrics (3 seeds, mean, n=150 per seed):

| Metric | causal_mcts | flow_ablation | random_floor | causal_mcts vs random_floor |
|---|---|---|---|---|
| BLEU | 0.313 | 0.281 | 0.317 | -1.3% |
| ROUGE-L | 0.096 | 0.095 | 0.095 | +1.1% |
| BERTScore F1 | 0.707 | 0.705 | 0.707 | +0.0% |
| Distinct-2 | 0.464 | 0.434 | 0.465 | -0.2% |
| Strategy accuracy | 0.127 | 0.120 | 0.178 | **-28.7%, significant (p=0.021)** |

Latency comparison: causal_mcts is about 37x faster per decision than an LLM-rollout approach (0.230s vs 8.590s).

2026-09-13 update: fixed a missing policy-loss term and a bug that pinned
every real state to ESC phase 0 (both detailed in `results/README.md`).
The global single-strategy mode collapse from the original pilot (39-150/150
depending on run) is gone -- `causal_mcts` now predicts 3 different,
phase-appropriate strategies -- but strategy accuracy did not improve, and
`causal_mcts` is now significantly worse than random guessing among
phase-legal strategies. See `results/README.md` for the full honest
writeup, including why (representational bottleneck, not the two bugs
fixed here).
