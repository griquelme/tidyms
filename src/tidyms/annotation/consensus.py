"""Implementation of the consensus annotation algorithm."""

from __future__ import annotations

from typing import Generator

import pydantic

from ..core.models import Annotation, GroupAnnotation


class MissingNodeError(ValueError):
    """Exception raised when a missing node group if requested to an annotation graph."""


class AnnotationGraph:
    """Store consensus annotation nodes and edges.

    A node represents a feature group, which stores charge status and isotopologue
    indices annotated on each assay sample.

    An edge represent a relation between an :term:`MMI` and another isotopologue
    in the group. It also stores the number of samples where the annotation was
    detected.

    """

    def __init__(self):
        self.nodes: dict[int, AnnotationNode] = dict()
        self.edges: dict[int, dict[int, int]] = dict()

    def __iter__(self) -> Generator[AnnotationNode, None, None]:
        for node in self.nodes.values():
            yield node

    def add_annotation(self, annotation: Annotation) -> None:
        """Add a new node or update an existing node using a feature annotation."""
        if annotation.group in self.nodes:
            node = self.nodes[annotation.group]
        else:
            node = AnnotationNode(group=annotation.group)
            self.nodes[annotation.group] = node

        node.charge_votes.setdefault(annotation.charge, 0)
        node.charge_votes[annotation.charge] += 1
        node.index_votes.setdefault(annotation.isotopologue_index, 0)
        node.index_votes[annotation.isotopologue_index] += 1

    def add_edge(self, group1: int, group2: int) -> None:
        """Create a new edge or update the edge count."""
        node1 = self.get_node(group1)
        node2 = self.get_node(group2)
        node1.neighbors.add(group2)
        node2.neighbors.add(group1)

        group1_edges = self.edges.setdefault(group1, dict())
        group1_edges.setdefault(group2, 0)
        group1_edges[group2] += 1

        group2_edges = self.edges.setdefault(group2, dict())
        group2_edges.setdefault(group1, 0)
        group2_edges[group1] += 1

    def get_edges(self, group: int) -> list[AnnotationEdge]:
        """Get node edges."""
        edges = list()
        node = self.get_node(group)
        node_edges_dict = self.edges.get(group, dict())
        for neighbor_group, edge_votes in node_edges_dict.items():
            neighbor = self.get_node(neighbor_group)
            edge = AnnotationEdge(groups=(node.group, neighbor.group), index=neighbor.index, votes=edge_votes)
            edges.append(edge)
        return edges

    def get_neighbors(self, group: int) -> list[AnnotationNode]:
        """Retrieve all neighbor nodes."""
        return [self.get_node(x) for x in self.get_node(group).neighbors]

    def get_node(self, group: int) -> AnnotationNode:
        """Fetch a node."""
        try:
            return self.nodes[group]
        except KeyError as e:
            msg = f"Node group {group} not found."
            raise MissingNodeError(msg) from e

    def list_node_groups(self) -> list[int]:
        """Retrieve the unique node groups in the graph."""
        return list(self.nodes)

    def remove_isolated(self) -> None:
        """Remove nodes with no neighbors."""
        for group in self.list_node_groups():
            node = self.get_node(group)
            if not node.neighbors:
                self.remove_node(node.group)

    def remove_node(self, group: int) -> None:
        """Remove a node if it exists."""
        edges = self.get_edges(group)
        for edge in edges:
            self.remove_edge(*edge.groups)
        self.nodes.pop(group)

    def remove_edge(self, group1: int, group2: int) -> None:
        """Remove an edge if it exists."""
        node1 = self.get_node(group1)
        node2 = self.get_node(group2)

        if group2 in node1.neighbors and group1 in node2.neighbors:
            self.edges[group1].pop(group2)
            self.edges[group2].pop(group1)

            node1.neighbors.remove(group2)
            node2.neighbors.remove(group1)
        else:
            msg = f"There is no edge between annotation node {group1} and {group2}."
            raise ValueError(msg)

    def count_votes(self) -> None:
        """Update node index and charge based on total counts."""
        for node in self:
            node.charge = max(node.charge_votes, key=lambda x: node.charge_votes[x])
            node.index = max(node.index_votes, key=lambda x: node.index_votes[x])


class AnnotationNode(pydantic.BaseModel):
    """Store a node with annotations."""

    group: int
    """The node id, equal to the feature group."""

    charge_votes: dict[int, int] = dict()
    """Maps charge state to number of feature occurrences."""

    index_votes: dict[int, int] = dict()
    """Maps isotopologue index to number of feature occurrences."""

    charge: int = -1
    """The most voted charge state"""

    index: int = -1
    """The most voted isotopologue index"""

    neighbors: set[int] = set()
    """The node neighbors"""


class AnnotationEdge(pydantic.BaseModel):
    """Store a node with annotations."""

    groups: tuple[int, int]
    """The pair of groups connected by the edge."""

    index: int
    """The isotopologue index that connects with the :term:`MMI`."""

    votes: int
    """The edge votes."""

    approval: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    """The approval metric for the annotation edge."""


def create_consensus_annotation(annotations: list[Annotation]) -> dict[int, GroupAnnotation]:
    """Create the annotations for all feature groups."""
    graph = create_annotation_graph(annotations)
    consensus = dict()

    # Create group annotations on groups of isotopologues to provide a common group id
    all_mmi = {x.group for x in graph if x.index == 0}
    isotopologue_group_id = 0
    while all_mmi:
        mmi = graph.get_node(all_mmi.pop())
        isotopologues_group = graph.get_neighbors(mmi.group)
        isotopologues_group.append(mmi)
        for isotopologues in trim_isotopologues(isotopologues_group):
            consensus[isotopologues.group] = GroupAnnotation(
                id=isotopologues.group,
                isotopologue_group=isotopologue_group_id,
                isotopologue_index=isotopologues.index,
                charge=isotopologues.charge,
            )
        isotopologue_group_id += 1
    return consensus


def create_annotation_graph(annotations: list[Annotation]) -> AnnotationGraph:
    """Create an annotation graph and add all annotations nodes and edges."""
    graph = AnnotationGraph()
    for sample_annotations in group_annotations_by_sample(annotations).values():
        add_sample_annotations(graph, sample_annotations)

    graph.count_votes()

    for node in graph:
        if node.index == 0:
            remove_invalid_mmi_neighbors(graph, node)

    for node in graph:
        if node.index == 0:
            solve_mmi_conflict(graph, node)
        else:
            solve_non_mmi_conflict(graph, node)

    graph.remove_isolated()

    return graph


def add_sample_annotations(graph: AnnotationGraph, annotations: list[Annotation]):
    """Add annotations from a single sample."""
    isotopologues_dict: dict[int, list[int]] = dict()
    mmi_dict: dict[int, int] = dict()

    for ann in annotations:
        if ann.group == -1 or ann.isotopologue_label == -1:
            continue
        graph.add_annotation(ann)
        if ann.isotopologue_index == 0:
            mmi_dict[ann.isotopologue_label] = ann.group
        else:
            isotopologues = isotopologues_dict.setdefault(ann.isotopologue_label, list())
            isotopologues.append(ann.group)

    for mmi_label, mmi_group in mmi_dict.items():
        isotopologues = isotopologues_dict.get(mmi_label)
        if isotopologues is None:
            continue

        for i in isotopologues:
            graph.add_edge(mmi_group, i)


def group_annotations_by_sample(annotations: list[Annotation]) -> dict[str, list[Annotation]]:
    """Group annotations by sample."""
    sample_to_annotations = dict()
    for ann in annotations:
        sample_annotations = sample_to_annotations.setdefault(ann.sample_id, list())
        sample_annotations.append(ann)
    return sample_to_annotations


def remove_invalid_mmi_neighbors(graph: AnnotationGraph, mmi: AnnotationNode):
    """Remove MMI neighbors with different charge state or equal indices."""
    rm_list = list()
    for neighbor_id in mmi.neighbors:
        neighbor = graph.get_node(neighbor_id)
        has_different_charge = neighbor.charge != mmi.charge
        has_equal_index = neighbor.index == mmi.index
        if has_different_charge or has_equal_index:
            rm_list.append(neighbor.group)

    for rm_group in rm_list:
        graph.remove_edge(mmi.group, rm_group)


def solve_mmi_conflict(graph: AnnotationGraph, mmi: AnnotationNode):
    """
    Remove neighbors with repeated index.

    Only the most voted is kept.
    """
    edges = graph.get_edges(mmi.group)
    index_to_edge: dict[int, list[AnnotationEdge]] = dict()
    for edge in edges:
        i_edges = index_to_edge.setdefault(edge.index, list())
        i_edges.append(edge)

    for i_edges in index_to_edge.values():
        most_voted_edge = max(i_edges, key=lambda x: x.votes)
        for edge in i_edges:
            if edge != most_voted_edge:
                graph.remove_edge(*edge.groups)


def solve_non_mmi_conflict(graph: AnnotationGraph, node: AnnotationNode) -> None:
    """Keep only the most voted neighbor for non-MMI nodes."""
    edges = graph.get_edges(node.group)

    if len(edges) > 1:
        most_voted_edge = max(edges, key=lambda x: x.votes)
        for edge in edges:
            if edge != most_voted_edge:
                graph.remove_edge(*edge.groups)


def compute_approval(edges: list[AnnotationEdge]) -> None:
    """Compute an edge approval metric."""
    # TODO: complete
    total = sum(x.votes for x in edges)
    for edge in edges:
        edge.approval = edge.votes / total


def trim_isotopologues(isotopologues: list[AnnotationNode]) -> list[AnnotationNode]:
    """
    Trim invalid feature annotations.

    Annotations are trimmed in the following cases:

    - isotopologue index 0 is not part of the index list in each isotopologue group
    - there are missing indices, i.e., 0, 1, 3, 4, ...

    """
    trimmed = list()
    for k, node in enumerate(sorted(isotopologues, key=lambda x: x.index)):
        if node.index != k:
            break
        trimmed.append(node)
    return trimmed
