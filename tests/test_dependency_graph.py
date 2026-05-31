"""Tests for prompt dependency graph."""

import pytest

from promptops.graph.dependency_graph import (
    PromptDependencyGraph,
    CircularDependencyError,
    PromptNode,
)


@pytest.fixture
def graph():
    """Create a fresh dependency graph."""
    return PromptDependencyGraph()


@pytest.fixture
def populated_graph():
    """Create a graph with a typical prompt hierarchy."""
    g = PromptDependencyGraph()
    g.add_prompt("system", name="System Prompt")
    g.add_prompt("persona", name="Persona Prompt")
    g.add_prompt("task", name="Task Prompt")
    g.add_prompt("output", name="Output Format")
    # persona depends on system, task depends on persona, output depends on task
    g.add_dependency("persona", "system")
    g.add_dependency("task", "persona")
    g.add_dependency("output", "task")
    return g


class TestGraphConstruction:
    """Test adding and removing prompts and dependencies."""

    def test_add_prompt(self, graph):
        """Test adding a prompt node."""
        node = graph.add_prompt("system", name="System Prompt", version="2.0.0")
        assert node.prompt_id == "system"
        assert node.name == "System Prompt"
        assert node.version == "2.0.0"
        assert "system" in graph.nodes

    def test_add_duplicate_prompt_returns_existing(self, graph):
        """Test that adding a duplicate prompt returns the existing node."""
        node1 = graph.add_prompt("system", name="First")
        node2 = graph.add_prompt("system", name="Second")
        assert node1 is node2
        assert node1.name == "First"  # Original preserved

    def test_remove_prompt_cleans_edges(self, populated_graph):
        """Test that removing a prompt removes all its edges."""
        populated_graph.remove_prompt("persona")
        assert "persona" not in populated_graph.nodes
        assert "persona" not in populated_graph.get_dependencies("task")
        assert "persona" not in populated_graph.get_dependents("system")

    def test_add_dependency_between_existing_prompts(self, graph):
        """Test adding a dependency between two existing prompts."""
        graph.add_prompt("a")
        graph.add_prompt("b")
        graph.add_dependency("a", "b")
        assert "b" in graph.get_dependencies("a")
        assert "a" in graph.get_dependents("b")

    def test_add_dependency_nonexistent_prompt_raises(self, graph):
        """Test that adding a dependency with a missing prompt raises ValueError."""
        graph.add_prompt("a")
        with pytest.raises(ValueError, match="not found"):
            graph.add_dependency("a", "nonexistent")


class TestCircularDependencyDetection:
    """Test circular dependency detection."""

    def test_self_dependency_raises(self, graph):
        """Test that a self-referencing dependency raises."""
        graph.add_prompt("self_ref")
        with pytest.raises(CircularDependencyError):
            graph.add_dependency("self_ref", "self_ref")

    def test_direct_cycle_raises(self, graph):
        """Test that a direct A->B->A cycle is detected."""
        graph.add_prompt("a")
        graph.add_prompt("b")
        graph.add_dependency("a", "b")
        with pytest.raises(CircularDependencyError):
            graph.add_dependency("b", "a")

    def test_indirect_cycle_raises(self, graph):
        """Test that an indirect A->B->C->A cycle is detected."""
        graph.add_prompt("a")
        graph.add_prompt("b")
        graph.add_prompt("c")
        graph.add_dependency("a", "b")
        graph.add_dependency("b", "c")
        with pytest.raises(CircularDependencyError):
            graph.add_dependency("c", "a")

    def test_has_circular_dependency_returns_false_for_dag(self, populated_graph):
        """Test that a valid DAG reports no circular dependencies."""
        assert populated_graph.has_circular_dependency() is False


class TestTopologicalSort:
    """Test topological sort for resolution order."""

    def test_linear_chain_sort(self, populated_graph):
        """Test topological sort of a linear dependency chain."""
        order = populated_graph.topological_sort()
        # system must come before persona, persona before task, task before output
        assert order.index("system") < order.index("persona")
        assert order.index("persona") < order.index("task")
        assert order.index("task") < order.index("output")

    def test_diamond_dependency_sort(self, graph):
        """Test topological sort with diamond-shaped dependencies."""
        graph.add_prompt("base")
        graph.add_prompt("left")
        graph.add_prompt("right")
        graph.add_prompt("top")
        graph.add_dependency("left", "base")
        graph.add_dependency("right", "base")
        graph.add_dependency("top", "left")
        graph.add_dependency("top", "right")

        order = graph.topological_sort()
        assert order.index("base") < order.index("left")
        assert order.index("base") < order.index("right")
        assert order.index("left") < order.index("top")
        assert order.index("right") < order.index("top")

    def test_independent_nodes_all_included(self, graph):
        """Test that independent nodes are included in sort."""
        graph.add_prompt("isolated_a")
        graph.add_prompt("isolated_b")
        graph.add_prompt("isolated_c")

        order = graph.topological_sort()
        assert set(order) == {"isolated_a", "isolated_b", "isolated_c"}


class TestTransitiveDependencies:
    """Test transitive dependency resolution."""

    def test_get_all_dependencies(self, populated_graph):
        """Test getting all transitive dependencies."""
        all_deps = populated_graph.get_all_dependencies("output")
        assert all_deps == {"task", "persona", "system"}

    def test_leaf_node_has_no_dependencies(self, populated_graph):
        """Test that a leaf node has no transitive dependencies."""
        all_deps = populated_graph.get_all_dependencies("system")
        assert all_deps == set()


class TestDOTExport:
    """Test DOT format visualization export."""

    def test_export_dot_format(self, populated_graph):
        """Test that DOT export produces valid format."""
        dot = populated_graph.export_dot(title="Test Graph")
        assert 'digraph "Test Graph"' in dot
        assert '"system"' in dot
        assert '"persona" -> "system"' in dot
        assert dot.strip().endswith("}")

    def test_empty_graph_exports_valid_dot(self, graph):
        """Test that an empty graph produces valid DOT."""
        dot = graph.export_dot()
        assert "digraph" in dot
        assert dot.strip().endswith("}")

    def test_dot_includes_node_labels(self, graph):
        """Test that DOT export includes node names and versions."""
        graph.add_prompt("test", name="Test Prompt", version="3.0.0")
        dot = graph.export_dot()
        assert "Test Prompt" in dot
        assert "v3.0.0" in dot
