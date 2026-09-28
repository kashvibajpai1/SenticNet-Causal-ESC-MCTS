"""
Real Qwen dialogue backbone (encoding + response generation).

Backbone: Qwen2.5-0.5B-Instruct — downsized from the paper's originally
stated "Qwen-9B". This machine has no GPU, so a 9B model is not feasible
here; 0.5B is the smallest Instruct model in the Qwen2.5 family and keeps
end-to-end CPU wall-clock reasonable. See results/README.md for the full
disclosure of this substitution and its implications.

**Integration point:** this module is the only place HF weights are
touched. All other ESC/MCTS/training code consumes
``QwenBackbone.encode_dialogue`` / ``QwenBackbone.generate_response`` and
stays agnostic to HF internals. The backbone weights are frozen and never
fine-tuned anywhere in this project — only the small policy/value/
transition MLPs downstream of it are trained.

Dimension bridging
-------------------
Qwen2.5-0.5B-Instruct's hidden size (896) does not match the fixed
``ESCState`` dimensions (D_H=768, D_C=384, D_E=128) hardcoded across
``PolicyNetwork``/``ValueNetwork``/``LinearTransitionModel``. Rather than
changing those dimensions everywhere, real Qwen hidden states are
projected down with FROZEN, seeded random linear projections
(Johnson-Lindenstrauss-style). These are not trained — a fixed,
reproducible dimensionality reduction of real semantic features. This is
a known, disclosed limitation, not a hidden one.

Cause-span heuristic
---------------------
ESConv has no ground-truth labels for which dialogue spans are "causes"
of distress, and the ``EncoderFn`` contract (``encoder(turns) -> dict``)
only receives raw turn strings, not corpus metadata. ``encode_dialogue``
therefore treats the seeker's turns (even indices — conversations are
validated to start with a seeker turn) as candidate distress-cause spans,
most recent first, capped at ``ESCState.N_C``. This is a real,
text-grounded heuristic, not a principled cause-extraction model —
documented as a limitation in results/README.md.
"""

from __future__ import annotations

import threading

import torch
from torch import nn

from esc.state import ESCState

_LOAD_LOCK = threading.Lock()


def _frozen_projection(in_dim: int, out_dim: int, seed: int) -> nn.Linear:
    """Seeded, frozen random linear projection (JL-style dimensionality reduction)."""
    gen = torch.Generator().manual_seed(seed)
    layer = nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        w = torch.randn(out_dim, in_dim, generator=gen) / (in_dim**0.5)
        layer.weight.copy_(w)
    for p in layer.parameters():
        p.requires_grad_(False)
    layer.eval()
    return layer


class QwenBackbone(nn.Module):
    """
    Real HF Qwen2.5-Instruct dialogue backbone (lazy-loaded, CPU, frozen weights).

    ``encode_dialogue`` returns tensors and cause dicts shaped for
    :meth:`esc.state.ESCState.from_dialogue` (history window, emotion, causes).
    ``generate_response`` produces a real greedy-decoded reply.
    """

    def __init__(self, model_name: str, *, seed: int = 42) -> None:
        super().__init__()
        self.model_name = model_name
        self._seed = seed
        self._tokenizer = None
        self._model = None
        self._hidden_size: int | None = None
        self._proj_history: nn.Linear | None = None
        self._proj_cause: nn.Linear | None = None
        self._proj_emotion: nn.Linear | None = None

    def load(self) -> None:
        """Load pretrained weights + tokenizer (idempotent, thread-safe, no-op after first call)."""
        if self._model is not None:
            return
        with _LOAD_LOCK:
            if self._model is not None:
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32
            )
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._hidden_size = int(self._model.config.hidden_size)
            self._proj_history = _frozen_projection(self._hidden_size, ESCState.D_H, self._seed)
            self._proj_cause = _frozen_projection(self._hidden_size, ESCState.D_C, self._seed + 1)
            self._proj_emotion = _frozen_projection(self._hidden_size, ESCState.D_E, self._seed + 2)

    def _pooled_embedding(self, text: str) -> torch.Tensor:
        """Mean-pool the final hidden state over non-pad tokens for one text span."""
        if not text.strip():
            return torch.zeros(self._hidden_size)
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0]  # [T, H]
        mask = enc["attention_mask"][0].unsqueeze(-1).float()  # [T, 1]
        summed = (last_hidden * mask).sum(dim=0)
        count = mask.sum().clamp(min=1.0)
        return (summed / count).float()

    def encode_dialogue(self, turns: list[str]) -> dict[str, torch.Tensor | list[dict]]:
        """
        Encode dialogue for ESC state construction with real Qwen embeddings.

        Returns
        -------
        dict with keys:

        - ``history`` : ``(ESCState.K_HISTORY_WINDOW, ESCState.D_H)`` — the
          most recent turns are encoded and placed in the last
          ``min(n, K)`` rows, zero-padded at the start when ``n < K``.
        - ``emotion`` : ``(ESCState.D_E,)`` — pooled embedding of the full
          dialogue so far, projected and squashed to ``(-1, 1)``.
        - ``causes``  : up to ``N_C`` dicts with ``"label"`` (a truncated
          text excerpt) and ``"embedding"`` (projected pooled span embedding).
        """
        self.load()
        k, d_h = ESCState.K_HISTORY_WINDOW, ESCState.D_H
        n_c = ESCState.N_C

        n = len(turns)
        history = torch.zeros(k, d_h)
        recent = turns[-k:] if n > 0 else []
        start_row = k - len(recent)
        for i, turn_text in enumerate(recent):
            pooled = self._pooled_embedding(turn_text)
            history[start_row + i] = self._proj_history(pooled)

        full_text = " ".join(turns) if turns else ""
        emotion_raw = self._pooled_embedding(full_text)
        emotion = torch.tanh(self._proj_emotion(emotion_raw))

        seeker_turns = [t for i, t in enumerate(turns) if i % 2 == 0 and t.strip()]
        candidate_spans = list(reversed(seeker_turns))[:n_c]
        causes: list[dict] = []
        for span in candidate_spans:
            pooled = self._pooled_embedding(span)
            causes.append({"label": span[:60], "embedding": self._proj_cause(pooled)})

        return {"history": history, "emotion": emotion, "causes": causes}

    def generate_response(self, prompt: str) -> str:
        """Real greedy-decoded response from the loaded Qwen model."""
        self.load()
        messages = [
            {
                "role": "system",
                "content": "You are a supportive, empathetic listener providing emotional support.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen_tokens = out[0][enc["input_ids"].shape[1] :]
        text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()
