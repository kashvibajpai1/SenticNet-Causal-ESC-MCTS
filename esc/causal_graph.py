"""
Causal graph for ESC: tracks identified causes and their resolution state.

The causal graph G_t = (V, E, ρ_t) is the "causally-aware" component
referenced in the paper title.  It stores:
  - V: cause nodes (each with a natural-language label and embedding)
  - E: directed causal edges (cause_i → cause_j means i contributes to j)
  - ρ_t ∈ [0,1]^{n_c}: per-cause resolution probability at turn t

Only the graph structure and resolution tracking live here.  Embedding
computation is handled upstream (backbone encoder) and passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class CauseNode:
    """
    A single identified cause in the causal graph.

    Attributes:
        index:       Position in the cause_embeddings matrix (0-indexed)
        label:       Human-readable description, e.g. "job insecurity"
        embedding:   Dense representation in ℝ^{d_c}; zeros until encoder runs
        resolution:  Probability that this cause has been addressed, ∈ [0, 1]
    """
    index: int
    label: str
    embedding: torch.Tensor                          # [d_c]
    resolution: float = 0.0                          # starts unresolved

    def __repr__(self) -> str:
        return (
            f"CauseNode(idx={self.index}, label='{self.label}', "
            f"resolution={self.resolution:.3f})"
        )


class CausalGraph:
    """
    Directed causal graph G_t = (V, E, ρ_t) over identified causes.

    Responsibilities
    ----------------
    - Store cause nodes and directed edges between them.
    - Maintain per-cause resolution probabilities ρ_t.
    - Expose resolution_tensor() for reward computation (R_cause uses Δρ_t).
    - Expose cause_embedding_matrix() so ESCState can build C_t from the graph.

    Usage
    -----
    At the start of a dialogue episode, populate the graph from extracted
    causes (strings + embeddings).  During each env.step(), call
    update_resolution() with the new probabilities produced by the
    TransitionModel.

    Parameters
    ----------
    d_c : int
        Cause embedding dimension (must match ESCState.D_C).
    """

    def __init__(self, d_c: int = 384) -> None:
        self.d_c = d_c
        self._nodes: list[CauseNode] = []
        # Adjacency: edges[i] = list of cause indices that cause i contributes to
        self._edges: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_cause(
        self,
        label: str,
        embedding: Optional[torch.Tensor] = None,
        initial_resolution: float = 0.0,
    ) -> CauseNode:
        """
        Add a cause node to the graph.

        Args:
            label:              Human-readable cause description
            embedding:          ℝ^{d_c} vector; zeros if None (placeholder)
            initial_resolution: Starting resolution probability [0, 1]

        Returns:
            The newly created CauseNode
        """
        idx = len(self._nodes)
        if embedding is None:
            embedding = torch.zeros(self.d_c)
        if embedding.shape != (self.d_c,):
            raise ValueError(
                f"Cause embedding must have shape ({self.d_c},), "
                f"got {tuple(embedding.shape)}"
            )
        node = CauseNode(
            index=idx,
            label=label,
            embedding=embedding.clone(),
            resolution=float(initial_resolution),
        )
        self._nodes.append(node)
        self._edges[idx] = []
        return node

    def add_edge(self, src: int, dst: int) -> None:
        """
        Add a directed causal edge src → dst.

        Meaning: cause[src] contributes to / exacerbates cause[dst].

        Args:
            src: Source cause index
            dst: Destination cause index

        Raises:
            IndexError: If either index is out of range
            ValueError: If src == dst (self-loops disallowed)
        """
        n = len(self._nodes)
        if not (0 <= src < n and 0 <= dst < n):
            raise IndexError(
                f"Edge indices ({src}, {dst}) out of range for graph with {n} nodes"
            )
        if src == dst:
            raise ValueError("Self-loops are not permitted in the causal graph")
        if dst not in self._edges[src]:
            self._edges[src].append(dst)

    # ------------------------------------------------------------------
    # Resolution management
    # ------------------------------------------------------------------

    def update_resolution(self, new_probs: torch.Tensor) -> None:
        """
        Overwrite per-cause resolution probabilities.

        Called by ESCEnv.step() after the TransitionModel produces updated ρ.

        Args:
            new_probs: Tensor of shape [n_c] with values in [0, 1]

        Raises:
            ValueError: If length doesn't match number of cause nodes
        """
        n = len(self._nodes)
        if new_probs.shape != (n,):
            raise ValueError(
                f"Expected resolution tensor of shape ({n},), "
                f"got {tuple(new_probs.shape)}"
            )
        for i, node in enumerate(self._nodes):
            node.resolution = float(new_probs[i].clamp(0.0, 1.0).item())

    def resolution_tensor(self) -> torch.Tensor:
        """
        Return current resolution probabilities as a tensor.

        Returns:
            Tensor of shape [n_c], dtype float32, values in [0, 1]
        """
        if not self._nodes:
            return torch.zeros(0)
        return torch.tensor(
            [node.resolution for node in self._nodes], dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Embedding access
    # ------------------------------------------------------------------

    def cause_embedding_matrix(self, max_causes: int, d_c: int) -> torch.Tensor:
        """
        Build the C_t matrix for ESCState from current node embeddings.

        Pads with zeros if fewer than max_causes nodes exist.
        Truncates if more than max_causes nodes exist.

        Args:
            max_causes: Number of rows in the output matrix (N_C)
            d_c:        Embedding dimension per cause (D_C)

        Returns:
            Tensor of shape [max_causes, d_c]
        """
        matrix = torch.zeros(max_causes, d_c)
        for i, node in enumerate(self._nodes[:max_causes]):
            if node.embedding.shape == (d_c,):
                matrix[i] = node.embedding
        return matrix

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def num_causes(self) -> int:
        """Number of cause nodes currently in the graph."""
        return len(self._nodes)

    def get_node(self, index: int) -> CauseNode:
        """Return cause node by index."""
        return self._nodes[index]

    def get_edges(self, src: int) -> list[int]:
        """Return list of destination indices for edges from src."""
        return list(self._edges.get(src, []))

    def neighbours(self, index: int) -> list[CauseNode]:
        """Return cause nodes reachable from cause[index] via one edge."""
        return [self._nodes[dst] for dst in self._edges.get(index, [])]

    @classmethod
    def placeholder(cls, n_causes: int, d_c: int) -> "CausalGraph":
        """
        Build a zero-filled graph with n_causes anonymous nodes and no edges.

        Used by ESCState.from_dialogue() until a real encoder runs.

        Args:
            n_causes: Number of placeholder cause nodes
            d_c:      Cause embedding dimension

        Returns:
            CausalGraph with n_causes nodes, all resolution=0, no edges
        """
        g = cls(d_c=d_c)
        for i in range(n_causes):
            g.add_cause(label=f"cause_{i}", embedding=torch.zeros(d_c))
        return g

    def __repr__(self) -> str:
        edge_count = sum(len(v) for v in self._edges.values())
        return (
            f"CausalGraph(nodes={self.num_causes}, edges={edge_count}, "
            f"resolution={[f'{n.resolution:.2f}' for n in self._nodes]})"
        )
