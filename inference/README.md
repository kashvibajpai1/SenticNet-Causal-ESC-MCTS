# inference -- Interactive ESC Session

This package provides an interactive command-line interface for running live ESC conversations using the MCTS planner and the Qwen backbone stub. It is the human-facing integration point that ties together all other components into a single read-eval loop.

---

## Files

### `interactive_cli.py` -- Primary Entry Point

`run_interactive_session()` starts an interactive loop that:

1. Loads the merged config from `config/env.yaml` and `config/train_shared.yaml`.
2. Sets the global random seed.
3. Initializes a `LinearTransitionModel`, `ESCEnv`, `PolicyNetwork`, `ValueNetwork`, and `MCTS` planner.
4. Initializes a `QwenBackbone` stub.
5. Resets the environment to produce an initial `ESCState`.
6. Enters a loop:
   - Reads a user utterance from stdin.
   - Appends it to the running dialogue list.
   - Calls `mcts.search(state)` to select the best strategy-cause action.
   - Constructs a prompt from the user utterance and selected action.
   - Calls `backbone.generate_response(prompt)` to produce a reply.
   - Prints the reply.
   - Calls `env.step(state, action)` to advance the MDP state.
   - If `done` is True (horizon reached), resets the environment and continues the conversation.

The MCTS planner uses the `LinearTransitionModel` (untrained weights at this stage) rather than `RandomTransitionModel`. This means action selection is informed by the policy and value networks, though with random initialization the quality is equivalent to the random baseline until training has been run.

`main()` is a thin wrapper around `run_interactive_session()` to support `python -m inference.interactive_cli`.

### `interactivecli.py`

An earlier draft of the interactive CLI. The canonical entry point is `interactive_cli.py`. This file is kept for reference but is not imported by any other module.

---

## Usage

```bash
python -m inference.interactive_cli
```

The session starts immediately. Type any user utterance and press enter. Type `quit` or `exit` to end the session.

---

## Integration with the full pipeline

When the Qwen backbone is wired in, the session will produce real embeddings for each turn. Replace the `QwenBackbone` stub by implementing `encode_dialogue()` and `generate_response()` with the actual HF model. No other code changes are needed.

The backbone can also be used to construct a real encoder and pass it to `env.reset(initial_turns=dialogue)` via `ESCState.from_dialogue(turns, encoder=encoder_adapter(backbone))` to seed the initial state from the actual conversation embeddings rather than zeros.

---

## Current limitations

- The backbone stub returns placeholder zero embeddings and a fixed reply string. The MCTS planner runs correctly but the reply is not grounded in the actual strategy.
- The policy and value networks are randomly initialized unless pre-trained weights are loaded manually.
- No checkpoint loading is implemented yet. Adding `policy.load_state_dict(torch.load("path"))` before the loop is the minimal extension needed.
