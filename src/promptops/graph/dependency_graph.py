"""Prompt dependency graph for cross-prompt references.

Tracks which prompts reference others, detects circular dependencies,
provides topological sort for resolution order, and exports to DOT format
for visualization.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("promptops.graph")


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the prompt graph."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}")


@dataclass
class PromptNode:
    """Represents a prompt in the dependency graph.

    Attributes:
        prompt_id: Unique identifier for the prompt.
        name: Human-readable name.
        version: Version string for the prompt.
        metadata: Additional metadata about the prompt.
    """
    prompt_id: str
    name: str = ""
    version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.prompt_id)

    def __eq__(self, other):
        if isinstance(other, PromptNode):
            return self.prompt_id == other.prompt_id
        return NotImplemented


class PromptDependencyGraph:
    """Directed graph tracking prompt-to-prompt dependencies.

    Supports:
    - Adding/removing prompts and their dependencies
    - Circular dependency detection
    - Topological sort for resolution order
    - DOT format export for visualization
    - Querying dependents and dependencies

    Example:
        graph = PromptDependencyGraph()
        graph.add_prompt("system", name="System Prompt")
        graph.add_prompt("chat", name="Chat Prompt")
        graph.add_dependency("chat", "system")  # chat depends on system
        order = graph.topological_sort()  # ["system", "chat"]
    """

    def __init__(self):
        self._nodes: dict[str, PromptNode] = {}
        self._edges: dict[str, set[str]] = defaultdict(set)  # prompt_id -> set of dependency IDs
        self._reverse_edges: dict[str, set[str]] = defaultdict(set)  # prompt_id -> set of dependent IDs

    @property
    def nodes(self) -> dict[str, PromptNode]:
        """Return all nodes in the graph."""
        return dict(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return total number of edges (dependencies) in the graph."""
        return sum(len(deps) for deps in self._edges.values())

    def add_prompt(
        self,
        prompt_id: str,
        name: str = "",
        version: str = "1.0.0",
        metadata: dict | None = None,
    ) -> PromptNode:
        """Add a prompt node to the graph.

        Args:
            prompt_id: Unique identifier for the prompt.
            name: Human-readable name (defaults to prompt_id).
            version: Version string.
            metadata: Additional metadata.

        Returns:
            The created or existing PromptNode.
        """
        if prompt_id in self._nodes:
            return self._nodes[prompt_id]

        node = PromptNode(
            prompt_id=prompt_id,
            name=name or prompt_id,
            version=version,
            metadata=metadata or {},
        )
        self._nodes[prompt_id] = node
        return node

    def remove_prompt(self, prompt_id: str) -> None:
        """Remove a prompt and all its edges from the graph.

        Args:
            prompt_id: The prompt to remove.
        """
        if prompt_id not in self._nodes:
            return

        # Remove all edges from this node
        for dep_id in self._edges.get(prompt_id, set()).copy():
            self._reverse_edges[dep_id].discard(prompt_id)
        self._edges.pop(prompt_id, None)

        # Remove all edges to this node
        for dependent_id in self._reverse_edges.get(prompt_id, set()).copy():
            self._edges[dependent_id].discard(prompt_id)
        self._reverse_edges.pop(prompt_id, None)

        del self._nodes[prompt_id]

    def add_dependency(self, prompt_id: str, depends_on: str) -> None:
        """Add a dependency edge: prompt_id depends on depends_on.

        Args:
            prompt_id: The prompt that has the dependency.
            depends_on: The prompt being depended upon.

        Raises:
            ValueError: If either prompt doesn't exist in the graph.
            CircularDependencyError: If adding this edge would create a cycle.
        """
        if prompt_id not in self._nodes:
            raise ValueError(f"Prompt '{prompt_id}' not found in graph")
        if depends_on not in self._nodes:
            raise ValueError(f"Prompt '{depends_on}' not found in graph")

        if prompt_id == depends_on:
            raise CircularDependencyError([prompt_id, prompt_id])

        # Check if adding this edge would create a cycle
        if self._would_create_cycle(prompt_id, depends_on):
            cycle = self._find_cycle_path(prompt_id, depends_on)
            raise CircularDependencyError(cycle)

        self._edges[prompt_id].add(depends_on)
        self._reverse_edges[depends_on].add(prompt_id)

    def remove_dependency(self, prompt_id: str, depends_on: str) -> None:
        """Remove a dependency edge.

        Args:
            prompt_id: The prompt that has the dependency.
            depends_on: The prompt being depended upon.
        """
        self._edges[prompt_id].discard(depends_on)
        self._reverse_edges[depends_on].discard(prompt_id)

    def get_dependencies(self, prompt_id: str) -> set[str]:
        """Get all direct dependencies of a prompt.

        Args:
            prompt_id: The prompt to query.

        Returns:
            Set of prompt IDs that this prompt depends on.
        """
        return set(self._edges.get(prompt_id, set()))

    def get_dependents(self, prompt_id: str) -> set[str]:
        """Get all prompts that directly depend on this prompt.

        Args:
            prompt_id: The prompt to query.

        Returns:
            Set of prompt IDs that depend on this prompt.
        """
        return set(self._reverse_edges.get(prompt_id, set()))

    def get_all_dependencies(self, prompt_id: str) -> set[str]:
        """Get all transitive dependencies of a prompt (recursive).

        Args:
            prompt_id: The prompt to query.

        Returns:
            Set of all prompt IDs in the dependency chain.
        """
        visited = set()
        queue = deque(self._edges.get(prompt_id, set()))

        while queue:
            dep = queue.popleft()
            if dep not in visited:
                visited.add(dep)
                queue.extend(self._edges.get(dep, set()) - visited)

        return visited

    def topological_sort(self) -> list[str]:
        """Return prompts in dependency resolution order (Kahn's algorithm).

        Returns:
            List of prompt IDs in topological order (dependencies first).

        Raises:
            CircularDependencyError: If the graph contains a cycle.
        """
        # Calculate in-degrees
        in_degree = {node_id: 0 for node_id in self._nodes}
        for prompt_id, deps in self._edges.items():
            for dep in deps:
                # dep has an incoming edge from prompt_id in the dependency sense
                # but for topo sort, we want deps to come first
                pass

        # Build adjacency for topo sort: edge from A to B means A must come before B
        # If prompt_id depends on depends_on, then depends_on must come first
        adj: dict[str, set[str]] = defaultdict(set)
        in_deg: dict[str, int] = {node_id: 0 for node_id in self._nodes}

        for prompt_id, deps in self._edges.items():
            for dep in deps:
                adj[dep].add(prompt_id)  # dep -> prompt_id (dep comes first)
                in_deg[prompt_id] = in_deg.get(prompt_id, 0) + 1

        # Kahn's algorithm
        queue = deque([n for n, d in in_deg.items() if d == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj.get(node, set()):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._nodes):
            raise CircularDependencyError(["cycle detected in graph"])

        return result

    def has_circular_dependency(self) -> bool:
        """Check if the graph contains any circular dependencies.

        Returns:
            True if a cycle exists, False otherwise.
        """
        try:
            self.topological_sort()
            return False
        except CircularDependencyError:
            return True

    def export_dot(self, title: str = "Prompt Dependency Graph") -> str:
        """Export the graph in DOT format for Graphviz visualization.

        Args:
            title: Title for the graph.

        Returns:
            DOT format string.
        """
        lines = [f'digraph "{title}" {{']
        lines.append("    rankdir=LR;")
        lines.append('    node [shape=box, style=filled, fillcolor=lightblue];')
        lines.append("")

        # Add nodes
        for prompt_id, node in self._nodes.items():
            label = f"{node.name}\\nv{node.version}"
            lines.append(f'    "{prompt_id}" [label="{label}"];')

        lines.append("")

        # Add edges
        for prompt_id, deps in self._edges.items():
            for dep in sorted(deps):
                lines.append(f'    "{prompt_id}" -> "{dep}";')

        lines.append("}")
        return "\n".join(lines)

    def _would_create_cycle(self, from_id: str, to_id: str) -> bool:
        """Check if adding an edge from from_id to to_id would create a cycle."""
        # If there's already a path from to_id to from_id, adding from_id -> to_id creates a cycle
        visited = set()
        queue = deque([to_id])

        while queue:
            current = queue.popleft()
            if current == from_id:
                return True
            if current not in visited:
                visited.add(current)
                queue.extend(self._edges.get(current, set()) - visited)

        return False

    def _find_cycle_path(self, from_id: str, to_id: str) -> list[str]:
        """Find the cycle path that would be created by adding from_id -> to_id."""
        # BFS from to_id to find path back to from_id
        visited = set()
        parent = {to_id: None}
        queue = deque([to_id])

        while queue:
            current = queue.popleft()
            if current == from_id:
                # Reconstruct path
                path = [from_id]
                node = to_id
                while node is not None:
                    path.append(node)
                    node = parent.get(node)
                # The cycle is: from_id -> to_id -> ... -> from_id
                path_reversed = list(reversed(path))
                path_reversed.append(from_id)
                return path_reversed

            if current not in visited:
                visited.add(current)
                for dep in self._edges.get(current, set()):
                    if dep not in visited:
                        parent[dep] = current
                        queue.append(dep)

        return [from_id, to_id, from_id]
