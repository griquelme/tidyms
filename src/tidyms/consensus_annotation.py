import pandas as pd
import networkx as nx
from collections import Counter
from typing import Dict, Final, List, Tuple
from .core import constants as c

_COUNT: Final[str] = "count"
_TOTAL: Final[str] = "total"


def vote_annotations(
    feature_table: pd.DataFrame,
) -> Tuple[nx.Graph, Dict[int, Dict[str, int]]]:
    """
    Assigns an isotopologue group to features based on annotations on each sample.

    Parameters
    ----------
    feature_table : DataFrame

    Returns
    -------
    graph : Graph
    annotations : Dict[int, Dict[str, int]]]
        A mapping from feature label to a dictionary with the following keys:
        ``charge`` contains the envelope charge, ``envelope_label`` is a label
        shared by isotopologues, ``envelope_index`` is the position of the envelope

    """
    nodes = _build_graph_nodes(feature_table)
    edges = _build_graph_edges(feature_table)
    graph = _build_isotopologue_graph(nodes, edges)
    _solve_annotation_conflicts(graph)
    annotations = _get_annotations(graph)
    return graph, annotations


def _build_graph_nodes(feature_table: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """
    Creates the nodes for the voting graph.

    Auxiliary function to vote_annotations.

    """
    nodes = dict()
    ft_label = feature_table[c.LABEL].to_numpy()
    charge = feature_table[c.CHARGE].to_numpy()
    envelope_label = feature_table[c.ENVELOPE_LABEL].to_numpy()
    envelope_index = feature_table[c.ENVELOPE_INDEX].to_numpy()

    # count feature index and charge for each feature
    q_votes = "charge_votes"
    i_votes = "index_votes"
    iterator = zip(ft_label, charge, envelope_label, envelope_index)
    for k_ft_label, k_charge, k_envelope_label, k_envelope_index in iterator:
        if (k_envelope_index > -1) and (k_ft_label > -1):
            ft_node = nodes.setdefault(k_ft_label, dict())
            ft_charge_votes = ft_node.setdefault(q_votes, dict())
            ft_index_votes = ft_node.setdefault(i_votes, dict())
            ft_charge_votes.setdefault(k_charge, 0)
            ft_index_votes.setdefault(k_envelope_index, 0)
            ft_charge_votes[k_charge] += 1
            ft_index_votes[k_envelope_index] += 1

    # select most voted charge and envelope index
    for ft_node in nodes.values():
        charge = max(ft_node[q_votes], key=lambda x: ft_node[q_votes][x])
        index = max(ft_node[i_votes], key=lambda x: ft_node[i_votes][x])
        ft_node[c.CHARGE] = charge
        ft_node[c.ENVELOPE_INDEX] = index
        ft_node.pop(q_votes)
        ft_node.pop(i_votes)
    return nodes


def _build_graph_edges(feature_table: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Creates edges for the voting graph.

    Auxiliary function to vote_annotations.

    """
    edges = list()
    sample = feature_table[c.SAMPLE].to_numpy()
    ft_label = feature_table[c.LABEL].to_numpy()
    envelope_label = feature_table[c.ENVELOPE_LABEL].to_numpy()
    envelope_index = feature_table[c.ENVELOPE_INDEX].to_numpy()
    edges_dict = dict()
    iterator = zip(sample, ft_label, envelope_label, envelope_index)
    for k_sample, k_ft_label, k_envelope_label, k_envelope_index in iterator:
        if (k_envelope_label > -1) and (k_ft_label > -1):
            sample_dict = edges_dict.setdefault(k_sample, dict())
            label_dict = sample_dict.setdefault(k_envelope_label, dict())
            label_dict[k_envelope_index] = k_ft_label

    for k, d in edges_dict.items():
        for el, label_dict in d.items():
            if 0 in label_dict:
                l0 = label_dict[0]
                e = [(l0, x) for x in sorted(label_dict.values()) if l0 != x]
                edges.extend(e)
    return edges


def _build_isotopologue_graph(
    nodes: Dict[int, Dict[str, float]], edges: List[Tuple[int, int]]
) -> nx.Graph:
    """
    Creates the voting graph, using nodes and edges.

    Each node in the graph contains two attributes: charge and index. Each
    edge in the graph contains a count attribute that counts in how many samples
    the annotation associates the pair of features joined by the edge.

    Aux function to vote_annotations.

    """
    graph = nx.Graph()

    graph.add_nodes_from(nodes)
    nx.set_node_attributes(graph, nodes)

    edge_count = Counter(edges)
    edge_count = {k: {_COUNT: v} for k, v in edge_count.items()}
    edges = set(edges)
    graph.add_edges_from(edges)
    nx.set_edge_attributes(graph, edge_count)

    return graph


def _solve_annotation_conflicts(graph: nx.Graph):
    """
    Remove edges with non-compatible annotations. A greedy strategy is adopted:
    the most frequent annotation is always kept.

    Aux function to vote_annotations.

    """
    _remove_mismatched_charge(graph)
    edge_attributes = dict()
    for node in graph:
        index = graph.nodes.get(node)[c.ENVELOPE_INDEX]
        grouped_edges = _group_edges(graph, node)
        if grouped_edges:
            if index == 0:
                edge_group_attrs = _solve_conflict_mmi(graph, grouped_edges)
            else:
                edge_group_attrs = _solve_conflict_not_mmi(graph, grouped_edges)

            for e in edge_group_attrs:
                if e in edge_attributes:
                    edge_attributes[e][_TOTAL] += edge_group_attrs[e][_TOTAL]
                else:
                    edge_attributes[e] = edge_group_attrs[e]
    nx.set_edge_attributes(graph, edge_attributes)


def _remove_mismatched_charge(graph: nx.Graph):
    """
    Remove edges between a pair of features in cases where their assigned charge
    differs.

    Aux function to _solve_annotation_conflicts.

    """
    for node1 in graph:
        node1_charge = graph.nodes.get(node1)[c.CHARGE]
        remove_edges = list()
        for node2 in graph.neighbors(node1):
            node2_charge = graph.nodes.get(node2)[c.CHARGE]
            if node1_charge != node2_charge:
                remove_edges.append((node1, node2))
        graph.remove_edges_from(remove_edges)


def _group_edges(graph: nx.Graph, node: int) -> Dict[int, List[Tuple[int, int]]]:
    """
    Groups edges that connects a node with their neighbours based on the
    envelope_index of their neighbours. Edges are grouped into a dictionary that
    maps envelope index to a list of edges.

    Auxiliary function to _solve_annotation_conflicts.

    """
    groups = dict()
    for neighbor in graph.neighbors(node):
        neigh_attr = graph.nodes.get(neighbor)
        index = neigh_attr[c.ENVELOPE_INDEX]
        index_group = groups.setdefault(index, list())
        index_group.append((node, neighbor))
    return groups


def _solve_conflict_mmi(
    graph: nx.Graph, grouped_edges: Dict[int, List[Tuple[int, int]]]
):
    """
    Solve conflicts with a group of edges from an MMI node (envelope_index==0).

    If more than two neighbours have the same envelope index, the most occurring
    is kept. An attribute called total is added to the edge to keep track of
    non-compatible annotations.

    Aux function to _solve_annotation_conflicts.

    """
    # If a neighbour has envelope_index == 0 it added to the list of edges to remove
    remove_edges = grouped_edges.pop(0, list())
    edge_attributes = dict()  # stores the total count of valid edges
    for edges in grouped_edges.values():
        max_count = 0
        total_count = 0
        keep = -1
        for e in edges:
            edge_counts = graph.edges.get(e)[_COUNT]
            total_count += edge_counts
            if edge_counts > max_count:
                max_count = edge_counts
                if keep != -1:
                    remove_edges.append(keep)
                keep = e
            else:
                remove_edges.append(e)
        if (keep != -1) and total_count:
            edge_attributes[keep] = {_TOTAL: total_count}
    graph.remove_edges_from(remove_edges)
    return edge_attributes


def _solve_conflict_not_mmi(
    graph: nx.Graph, grouped_edges: Dict[int, List[Tuple[int, int]]]
):
    """
    Solve conflicts with a group of edges connecting to a non-MMI node.

    Neighbours with non-zero envelope_index are removed. If more than one MMI
    is a neighbour, the edge with the greatest count is kept.

    Aux function to _solve_annotation_conflicts.

    """
    remove_edges = list()
    edge_attributes = dict()
    for k, v in grouped_edges.items():
        if k != 0:
            remove_edges.extend(v)
    edges = grouped_edges.get(0, list())
    max_count = 0
    total_count = 0
    keep = -1
    for e in edges:
        edge_counts = graph.edges.get(e)[_COUNT]
        total_count += edge_counts
        if edge_counts > max_count:
            max_count = edge_counts
            if keep != -1:
                remove_edges.append(keep)
            keep = e
        else:
            remove_edges.append(e)
    if (keep != -1) and total_count:
        edge_attributes[keep] = {_TOTAL: total_count}
    graph.remove_edges_from(remove_edges)
    return edge_attributes


def _get_annotations(graph: nx.Graph) -> Dict[int, Dict[str, int]]:
    """
    Extracts the annotations from a graph. Annotations are stores as a dictionary
    where each key is a feature label, and its values are the envelope charge,
    envelope label and envelope index.

    Aux function to vote_annotations.

    """
    annotations = dict()
    label_counter = 0
    for nodes in nx.connected_components(graph):
        if len(nodes) > 1:
            for n in nodes:
                node_attributes = graph.nodes.get(n)
                node_index = node_attributes.get(c.ENVELOPE_INDEX)
                if node_index:
                    # non MMI nodes have only one edge
                    edge = list(graph.edges(n))[0]
                    edge_attributes = graph.edges.get(edge)
                    edge_count = edge_attributes[_COUNT]
                    edge_total = edge_attributes[_TOTAL]
                else:
                    edge_count = 0
                    edge_total = 0

                node_annotation = {
                    c.ENVELOPE_LABEL: label_counter,
                    c.CHARGE: node_attributes.get(c.CHARGE),
                    c.ENVELOPE_INDEX: node_index,
                    _TOTAL: edge_total,
                    _COUNT: edge_count,
                }
                annotations[n] = node_annotation
            label_counter += 1
    return annotations
