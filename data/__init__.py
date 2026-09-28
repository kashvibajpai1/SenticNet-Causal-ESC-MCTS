"""Dataset preprocessing → ESC states (offline).

Keep Hugging Face / ConvoKit imports out of training and MCTS entrypoints;
call ``dataset_sources`` only from preprocessing scripts or tests.
"""

from data.build_states import (
    build_esc_state_from_record,
    deterministic_target_emotion,
    encoder_adapter,
    extract_cause_labels,
)
from data.collate import collate_bundles, collate_state_tensors
from data.conversation_schema import ConversationRecord
from data.dataset_sources import (
    convokit_corpus_to_records,
    esconv_row_to_record,
    iter_cornell_esc_records,
    iter_esconv_records,
    load_cornell_esc_corpus,
)
from data.preprocess import normalize_strategy_label, preprocess_record, validate_record
from data.serialization import (
    causal_graph_from_bundle,
    causal_graph_to_bundle,
    esc_state_from_bundle,
    esc_state_to_bundle,
    iter_jsonl,
    load_state_split,
    save_state_split,
    write_jsonl,
)

__all__ = [
    "ConversationRecord",
    "build_esc_state_from_record",
    "collate_bundles",
    "collate_state_tensors",
    "convokit_corpus_to_records",
    "deterministic_target_emotion",
    "encoder_adapter",
    "esc_state_from_bundle",
    "esc_state_to_bundle",
    "causal_graph_from_bundle",
    "causal_graph_to_bundle",
    "extract_cause_labels",
    "esconv_row_to_record",
    "iter_cornell_esc_records",
    "iter_esconv_records",
    "load_cornell_esc_corpus",
    "iter_jsonl",
    "load_state_split",
    "normalize_strategy_label",
    "preprocess_record",
    "save_state_split",
    "validate_record",
    "write_jsonl",
]
