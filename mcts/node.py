"""
MCTS tree node with PUCT statistics.

Changes from v1
---------------
- select_child() is now correct when N=0: uses a safe sqrt that returns 0
  (not needed at root because root.N is set to 1 before selection begins,
  but defensive coding is better here).
- backpropagate() traverses the path from leaf to root updating all ancestors,
  rather than requiring the caller to manually walk the tree.
- Added depth property for debugging.
- visit_count_policy() returns a normalised distribution over children's
  visit counts — the standard output used to extract a policy target for
  training (used in the MCTS training loop).
"""

from __future__ import annotations

import math
from typing import Optional
import torch

from esc.state import ESCState
from esc.action import ESCAction


class TreeNode:
    """
    A node in the MCTS search tree.

    PUCT formula (Rosin 2011 / AlphaGo Zero variant):
        PUCT(s, a) = Q(s,a) + c · P(a|s) · √N(s) / (1 + N(s,a))

    Attributes
    ----------
    state         : ESCState at this node
    parent        : Parent TreeNode (None = root)
    parent_action : Action taken from parent to reach this node
    N : Visit count
    W : Total accumulated value
    Q : Mean value W/N
    P : Prior probability π_θ(a | s_parent)
    children      : {ESCAction → TreeNode}
    is_expanded   : True once expand() has been called
    """

    def __init__(
        self,
        state: ESCState,
        parent: Optional["TreeNode"] = None,
        parent_action: Optional[ESCAction] = None,
        prior: float = 1.0,
    ) -> None:
        self.state: ESCState = state
        self.parent: Optional[TreeNode] = parent
        self.parent_action: Optional[ESCAction] = parent_action

        # PUCT statistics
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = float(prior)

        # Children
        self.children: dict[ESCAction, TreeNode] = {}
        self.is_expanded: bool = False

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    def expand(
        self,
        actions: list[ESCAction],
        priors: torch.Tensor,
    ) -> None:
        """
        Create child nodes for each candidate action.

        Child states are clones of the current state.  In the full MCTS
        loop (MCTS._simulate), env.step() is called to get real next states
        before expansion, so the child states are set to the actual
        next state, not just a clone of the parent.

        Parameters
        ----------
        actions : Candidate ESCActions (from generate_candidate_actions)
        priors  : Policy probabilities π_θ(a|s), shape [len(actions)]

        Raises
        ------
        AssertionError : If len(actions) != priors.shape[0]
        RuntimeError   : If node is already expanded
        """
        if self.is_expanded:
            raise RuntimeError(
                "expand() called on an already-expanded node.  "
                "Each node should be expanded exactly once."
            )
        if len(actions) != priors.shape[0]:
            raise AssertionError(
                f"Length mismatch: {len(actions)} actions vs "
                f"{priors.shape[0]} priors"
            )

        for action, prior_prob in zip(actions, priors):
            child = TreeNode(
                state=self.state.clone(),
                parent=self,
                parent_action=action,
                prior=float(prior_prob.item()),
            )
            self.children[action] = child

        self.is_expanded = True

    def set_child_state(self, action: ESCAction, state: ESCState) -> None:
        """
        Override a child's state after environment simulation.

        Called by MCTS._simulate() to replace the placeholder clone with
        the real next state from env.step().

        Parameters
        ----------
        action : The action that leads to the child
        state  : The real next state from env.step()

        Raises
        ------
        KeyError : If action is not in self.children
        """
        if action not in self.children:
            raise KeyError(f"Action {action} not found in children")
        self.children[action].state = state

    def select_child(self, c_puct: float = 1.0) -> tuple[ESCAction, "TreeNode"]:
        """
        Select the child with the highest PUCT score.

        Handles the N=0 edge case: when the parent has never been visited
        (root initialised with N=0), sqrt(N)=0 so the U term is 0 for all
        children.  In that case selection falls back to the highest prior,
        which is correct behaviour (exploration is impossible without visits).

        Parameters
        ----------
        c_puct : Exploration constant (default 1.0)

        Returns
        -------
        (best_action, best_child_node)

        Raises
        ------
        AssertionError : If node has no children (call expand first)
        """
        assert self.children, (
            "select_child() called on a node with no children.  "
            "Call expand() first."
        )

        sqrt_n_parent = math.sqrt(max(self.N, 1))  # safe: ≥1 avoids 0 term

        best_action: Optional[ESCAction] = None
        best_child: Optional[TreeNode] = None
        best_value: float = -float("inf")

        for action, child in self.children.items():
            u = c_puct * child.P * sqrt_n_parent / (1.0 + child.N)
            puct = child.Q + u
            if puct > best_value:
                best_value = puct
                best_action = action
                best_child = child

        assert best_action is not None
        return best_action, best_child  # type: ignore[return-value]

    def update(self, value: float) -> None:
        """
        Update statistics at this node with a single value estimate.

        N ← N + 1
        W ← W + value
        Q ← W / N

        Parameters
        ----------
        value : Scalar value estimate from rollout or value network
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def backpropagate(self, value: float) -> None:
        """
        Update this node and all ancestors up to the root.

        Walks the parent chain, calling update() on each node.  The value
        is not negated between levels because ESC is a single-agent MDP
        (not a two-player zero-sum game), so the same value applies at
        every depth.

        Parameters
        ----------
        value : Value estimate to propagate (e.g. cumulative rollout reward
                or value-network output at the leaf)
        """
        node: Optional[TreeNode] = self
        while node is not None:
            node.update(value)
            node = node.parent

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def visit_count_policy(self, temperature: float = 1.0) -> tuple[list[ESCAction], torch.Tensor]:
        """
        Derive a policy distribution from child visit counts.

        Used to generate training targets for the policy network:
            π_target(a) ∝ N(s, a)^{1/temperature}

        temperature=1   → proportional to visit counts
        temperature→0   → one-hot on the most-visited action
        temperature→∞   → uniform

        Parameters
        ----------
        temperature : Softmax temperature τ > 0 (default 1.0)

        Returns
        -------
        (actions, probs) : list of ESCActions and corresponding probability tensor
        """
        if not self.children:
            return [], torch.zeros(0)

        actions = list(self.children.keys())
        counts = torch.tensor(
            [self.children[a].N for a in actions], dtype=torch.float32
        )

        if temperature == 0.0:
            # Greedy: one-hot on argmax
            probs = torch.zeros_like(counts)
            probs[int(counts.argmax().item())] = 1.0
        else:
            counts = counts ** (1.0 / temperature)
            probs = counts / (counts.sum() + 1e-8)

        return actions, probs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """True if this node has not been expanded yet."""
        return not self.is_expanded

    @property
    def depth(self) -> int:
        """Distance from the root (root.depth == 0)."""
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    def most_visited_child(self) -> tuple[ESCAction, "TreeNode"]:
        """
        Return the child with the highest visit count.

        Used at the end of search to extract the final action recommendation
        (visit count is more robust than Q for final selection).

        Raises
        ------
        AssertionError : If node has no children
        """
        assert self.children, "No children to select from"
        best_action = max(self.children, key=lambda a: self.children[a].N)
        return best_action, self.children[best_action]

    def __repr__(self) -> str:
        action_str = repr(self.parent_action) if self.parent_action else "root"
        return (
            f"TreeNode(action={action_str}, depth={self.depth}, "
            f"N={self.N}, Q={self.Q:.4f}, P={self.P:.4f})"
        )
