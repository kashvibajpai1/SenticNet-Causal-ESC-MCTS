# data -- Offline Dataset Layer

This package is responsible for the entire data pipeline from raw corpora to serialized `ESCState` objects. It is fully decoupled from the training, MCTS, and model layers: nothing in `esc/`, `mcts/`, `models/`, or `train/` imports from here. The dependency arrow points inward only -- data knows about `esc/` (to build states), but `esc/` does not know about data.

---

## Design philosophy

The pipeline follows a strict sequence of transformations:

```
Raw corpus (HF / ConvoKit)
  -> ConversationRecord   (conversation_schema.py)
  -> cleaned record       (preprocess.py)
  -> ESCState             (build_states.py)
  -> serialized artifact  (serialization.py)
  -> PyTorch Dataset      (train/train_data.py)
```

Each transformation lives in its own module. No module skips a stage. This makes it easy to inspect intermediate artifacts, replay any single step, and swap out one source without affecting the rest of the pipeline.

Heavy dataset dependencies (`datasets`, `convokit`) are imported lazily, inside the functions that need them. Importing `data.dataset_sources` at the top level never triggers a Hugging Face or ConvoKit download.

---

## Files

### `conversation_schema.py` -- ConversationRecord

`ConversationRecord` is the canonical intermediate representation that unifies the ESConv and Cornell ESC formats into a single structure before any encoding takes place.

```python
@dataclass
class ConversationRecord:
    conversation_id: str          # stable identifier (from source or synthesized)
    source:          str          # "esconv" or "cornell_esc"
    turns:           list[str]    # ordered utterance strings
    speaker_roles:   list[str]    # "seeker" or "supporter" per turn
    annotations:     dict         # turn_strategies, survey_score
    metadata:        dict         # split, problem_type, emotion_type, situation
```

`__post_init__` enforces that `turns` and `speaker_roles` have the same length. A mismatch at construction time is always a bug in the loader, so it raises immediately.

`to_json_obj()` and `from_json_obj()` provide the JSONL serialization contract. All fields are plain Python types (str, list, dict), making the serialized format human-readable and version-control friendly.

The `annotations` dict carries:
- `turn_strategies`: list of raw strategy labels, one per turn (None where unavailable)
- `turn_strategies_raw`: preserved original labels before normalization (added by preprocess.py)
- `survey_score`: end-of-conversation satisfaction score where available

The `metadata` dict carries:
- `split`: "train", "validation", or "test" (None for Cornell, which has no pre-defined splits)
- `problem_type`, `emotion_type`, `situation`: free-text context fields

---

### `dataset_sources.py` -- Raw Corpus Loaders

This module contains two loaders: one for ESConv (Hugging Face) and one for Cornell ESC (ConvoKit).

**ESConv loader**

`iter_esconv_records(split, max_samples, streaming)` is a generator that yields `ConversationRecord` objects from the `thu-coai/esconv` dataset on Hugging Face.

`esconv_row_to_record(row, split, index)` maps a single HF row to a record. The raw HF schema uses the speaker labels `"usr"` / `"sys"`, which are normalized to `"seeker"` / `"supporter"` by `_normalize_role_esconv()`. When a speaker label is missing or unrecognized, `_infer_roles_alternating()` fills it in by alternating from the last known role, seeding with `"seeker"` if no known role exists.

Strategy labels from ESConv are captured as-is in `annotations["turn_strategies"]`. They are normalized to the canonical `ESCAction.STRATEGIES` names later in `preprocess.py`.

**Cornell ESC loader**

`load_cornell_esc_corpus()` downloads and returns the ConvoKit `emotional-support` corpus. This is a separate function so that callers can fail fast if ConvoKit is not installed before starting the (much slower) ESConv iteration.

`iter_cornell_esc_records(max_samples)` wraps `convokit_corpus_to_records()`, which iterates over conversations in the corpus. Utterances within each conversation are sorted by a numeric suffix in their ID (`_utterance_sort_key`) to reconstruct chronological order, which ConvoKit does not guarantee.

Cornell speaker roles are read from `speaker.meta["role"]`. Any value that is not `"seeker"` or `"supporter"` is mapped to `"seeker"` as a conservative fallback.

Cornell strategy annotations are read from `utt.meta["annotation"]`, which may be a dict (with key `"strategy"` or `"strategies"`), a list, or a bare string. All three formats are handled.

---

### `preprocess.py` -- Deterministic Cleaning Pipeline

`preprocess_record(record, min_turns, max_turns, rng)` applies a sequence of deterministic transformations to a `ConversationRecord` and returns a cleaned record, or `None` if the record should be dropped.

**Steps in order:**

1. Strip each utterance of leading/trailing whitespace and collapse internal whitespace to single spaces.
2. Drop empty utterances (and their corresponding role and strategy entries).
3. Re-normalize roles: any role that is not `"seeker"` or `"supporter"` is mapped to `"seeker"`.
4. If the first turn is not from a seeker, rotate forward to the first seeker turn. This ensures all records start with the user's disclosure.
5. Drop the record if it has fewer than `min_turns` (default 2) turns after cleaning.
6. Truncate to `max_turns` (default 60) turns.
7. Normalize strategy labels via `normalize_strategy_label()`.

`normalize_strategy_label(raw)` maps raw corpus-specific strategy strings to the canonical `ESCAction.STRATEGIES` names. It uses an exact-match dict `_RAW_TO_INTERNAL` first, then falls back to a substring/contains match for variants like `"Affirmation / Reassurance"`. Labels that cannot be mapped return `None`.

`validate_record(record, min_turns)` is a lightweight post-hoc check that can be run on any record (preprocessed or otherwise) to confirm it meets the minimum requirements before persistence:
- At least `min_turns` turns
- Source is `"esconv"` or `"cornell_esc"`
- Non-empty conversation ID
- First speaker is `"seeker"`

---

### `build_states.py` -- ConversationRecord to ESCState

This module bridges the data layer and the MDP layer. It takes a preprocessed `ConversationRecord` and produces an `ESCState` with correctly shaped tensors.

**`build_esc_state_from_record(record, encoder, target_emotion)`**

When `encoder` is `None`, this calls `ESCState.from_dialogue(turns)` and returns a structurally valid zero-filled state. This is the path used for smoke testing without a live backbone.

When an `encoder` is provided (via `encoder_adapter()`), it is passed through to `ESCState.from_dialogue(turns, encoder=encoder)`. The encoder receives the full turn list and returns `history`, `emotion`, and `causes` tensors.

The `target_emotion` is derived from the record's `metadata["emotion_type"]` field via `deterministic_target_emotion()`. This function uses SHA256 of the emotion type string to produce a stable, bounded vector in [-1, 1]^{D_E}. The SHA256 approach guarantees:
- Determinism across Python versions and processes
- Different emotion types map to different vectors (with high probability)
- The same emotion type always maps to the same vector

When `emotion_type` is absent or empty, `target_emotion` defaults to the zero vector (calm/neutral).

Two provenance attributes are attached to the resulting state as non-MDP metadata: `_conversation_id` and `_source`. These are not part of the tensor contract and are used only for debugging and logging.

**`extract_cause_labels(record)`** produces a short list of cause label strings from the record's metadata (`problem_type`, `situation`). If neither field is populated, it returns `["unspecified_distress"]`. This provides at least one meaningful cause label for graph initialization even when the corpus lacks explicit cause annotations.

**`encoder_adapter(backbone)`** wraps a `QwenBackbone`-like object into the `EncoderFn` callable type. The adapter calls `backbone.encode_dialogue(turns)` and forwards the result transparently.

---

### `serialization.py` -- Artifact I/O

This module handles all persistence for the two artifact types produced by the preprocessing pipeline: JSONL conversation records and `.pt` state bundles.

**JSONL artifacts (`write_jsonl`, `iter_jsonl`)**

`write_jsonl(path, records)` writes one JSON object per line to `artifacts/processed/conversations.jsonl`. The parent directory is created if it does not exist. It returns the count of records written.

`iter_jsonl(path)` is a generator that reads back and reconstructs `ConversationRecord` objects from a JSONL file. It skips blank lines.

**State bundle artifacts (`save_state_split`, `load_state_split`)**

A state bundle is a Python dict saved with `torch.save`. It contains:
- `"bundles"`: a list of per-state dicts, each with all ESCState tensor fields and causal graph data
- `"state_tensors"`: an optional pre-stacked `[N, state_dim]` tensor for fast loading in `ESCStateTensorDataset`

`esc_state_to_bundle(state)` converts a single ESCState to a serializable dict. All tensors are detached and moved to CPU before saving.

`esc_state_from_bundle(bundle)` reconstructs an ESCState from a bundle dict. It also restores the `_conversation_id` and `_source` provenance attributes.

`causal_graph_to_bundle(graph)` and `causal_graph_from_bundle(bundle)` handle the causal graph separately, storing node labels, embeddings as a stacked tensor, resolution values, and edge pairs as a list of `[src, dst]` pairs.

---

### `collate.py` -- DataLoader Collate Functions

`collate_state_tensors(batch)` stacks a list of flat state tensors into a `[B, state_dim]` batch tensor. This is passed as the `collate_fn` to `torch.utils.data.DataLoader` in `scripts/run_flow_ablation.py`.

`collate_bundles(batch)` is a passthrough that returns the list of bundle dicts unchanged. It is used when the training loop needs full `ESCState` objects rather than pre-flattened tensors (as in `CausalMCTSTrainer`).

---

## Artifact layout

After running `python -m scripts.preprocess_datasets`, the following files are created under `artifacts/` (gitignored):

```
artifacts/
  processed/
    conversations.jsonl       # all cleaned ConversationRecords, one per line
  states/
    train.pt                  # {bundles: [...], state_tensors: Tensor[N_train, 1283]}
    valid.pt
    test.pt
```

The `.pt` files are split by the `"split"` field in each record's metadata. Cornell ESC records (which have no pre-defined split) are assigned to train/valid/test by a deterministic hash of the conversation ID.

---

## Dependency isolation

| Module | Allowed imports |
|---|---|
| `conversation_schema.py` | stdlib only |
| `dataset_sources.py` | `data.conversation_schema`, lazy `datasets`/`convokit` |
| `preprocess.py` | `data.conversation_schema`, `esc.action` (for strategy names) |
| `build_states.py` | `data.conversation_schema`, `esc.state` |
| `serialization.py` | `data.conversation_schema`, `esc.state`, `esc.causal_graph`, `torch` |
| `collate.py` | `torch` only |

None of these modules import from `train/`, `mcts/`, `models/`, `flow/`, or `scripts/`.
