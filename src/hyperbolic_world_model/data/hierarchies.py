"""Build embodiment > task > primitive trees from action metadata.

Each trajectory carries labels ``{"embodiment": ..., "task": ..., "primitive": ...}``. We build the
rooted tree ``root -> embodiment -> task -> primitive`` and expose its hop-count metric and
adjacency, which are the ground truth for hierarchy reconstruction (``metrics/distortion.py``)
and a 0-hyperbolic reference for the Gromov estimator tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

LEVELS: tuple[str, ...] = ("embodiment", "task", "primitive")


@dataclass
class Hierarchy:
    """A rooted tree stored as parent pointers.

    Attributes:
        names: node label per index; index ``0`` is the root ``"<root>"``.
        parent: parent index per node; ``-1`` for the root.
        depth: depth per node (root ``0``).
        levels: level name per node (``"root"``, then ``level_names`` by depth).
        level_names: the level name of each depth below the root (defaults to :data:`LEVELS`).
    """

    names: list[str] = field(default_factory=lambda: ["<root>"])
    parent: list[int] = field(default_factory=lambda: [-1])
    depth: list[int] = field(default_factory=lambda: [0])
    levels: list[str] = field(default_factory=lambda: ["root"])
    level_names: tuple[str, ...] = LEVELS
    _index: dict[tuple[str, ...], int] = field(default_factory=lambda: {(): 0}, repr=False)

    # ------------------------------------------------------------------ construction
    def add_path(self, path: Sequence[str]) -> int:
        """Insert ``path`` (e.g. ``("arm_a", "task_1", "grasp")``); returns the leaf index."""
        node = 0
        for lvl, label in enumerate(path):
            key = tuple(path[: lvl + 1])
            if key not in self._index:
                self.names.append(label)
                self.parent.append(node)
                self.depth.append(lvl + 1)
                self.levels.append(
                    self.level_names[lvl] if lvl < len(self.level_names) else f"level_{lvl}"
                )
                self._index[key] = len(self.names) - 1
            node = self._index[key]
        return node

    def index_of(self, path: Sequence[str]) -> int:
        return self._index[tuple(path)]

    @property
    def n_nodes(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------ metric structure
    def adjacency(self) -> Tensor:
        """Symmetric boolean ``(n, n)`` edge matrix."""
        n = self.n_nodes
        adj = torch.zeros(n, n, dtype=torch.bool)
        for child, par in enumerate(self.parent):
            if par >= 0:
                adj[child, par] = adj[par, child] = True
        return adj

    def ancestors(self, node: int) -> list[int]:
        """Path from ``node`` up to and including the root."""
        out = [node]
        while self.parent[out[-1]] >= 0:
            out.append(self.parent[out[-1]])
        return out

    def lca_depth(self, a: int, b: int) -> int:
        anc_a = set(self.ancestors(a))
        for node in self.ancestors(b):
            if node in anc_a:
                return self.depth[node]
        raise RuntimeError("nodes do not share a root")  # unreachable for a tree

    def ancestor_matrix(self) -> Tensor:
        """``(n, max_depth + 1)`` root-to-node paths as node ids, padded with ``-1``."""
        n = self.n_nodes
        width = max(self.depth) + 1
        mat = torch.full((n, width), -1, dtype=torch.long)
        for node in range(n):
            path = self.ancestors(node)[::-1]  # root first
            mat[node, : len(path)] = torch.tensor(path)
        return mat

    def tree_distance_matrix(self) -> Tensor:
        """Hop-count distance ``(n, n)`` via lowest common ancestors; a 0-hyperbolic metric.

        Vectorised: the LCA depth of two nodes is the length of the common prefix of their
        root-to-node paths minus one, computed for all pairs at once on the ancestor matrix.
        The previous pairwise Python loop took 14 s for 1.5k nodes.
        """
        paths = self.ancestor_matrix()  # (n, w)
        same = (paths.unsqueeze(1) == paths.unsqueeze(0)) & (paths.unsqueeze(1) >= 0)  # (n, n, w)
        common = torch.cumprod(same.to(torch.int64), dim=-1).sum(
            -1
        )  # common prefix length (>= 1: the root)
        depth = torch.tensor(self.depth, dtype=torch.int64)
        d = depth.unsqueeze(1) + depth.unsqueeze(0) - 2 * (common - 1)
        return d.to(torch.float64)

    def leaves(self) -> list[int]:
        children = set(p for p in self.parent if p >= 0)
        return [i for i in range(self.n_nodes) if i not in children]


def build_hierarchy(
    metadata: Iterable[Mapping[str, str]], levels: Sequence[str] = LEVELS
) -> Hierarchy:
    """Build the tree from per-trajectory metadata dicts.

    Args:
        metadata: iterable of dicts containing every key in ``levels``.
        levels: ordered keys, outermost first. Defaults to embodiment > task > primitive.

    Raises:
        KeyError: if a record is missing a level key.
    """
    tree = Hierarchy(level_names=tuple(str(k) for k in levels))
    for rec in metadata:
        tree.add_path([str(rec[k]) for k in levels])
    return tree


def leaf_index_for_metadata(
    tree: Hierarchy, metadata: Iterable[Mapping[str, str]], levels: Sequence[str] = LEVELS
) -> Tensor:
    """Map each metadata record to its leaf node index (for aligning latents with tree nodes)."""
    return torch.tensor(
        [tree.index_of([str(rec[k]) for k in levels]) for rec in metadata], dtype=torch.long
    )


__all__ = ["LEVELS", "Hierarchy", "build_hierarchy", "leaf_index_for_metadata"]
