import pytest
from tidyms.annotation import consensus
from tidyms.annotation.consensus import AnnotationGraph, MissingNodeError
from tidyms.core.models import Annotation


class TestAnnotationGraphNodeFunctionality:
    @pytest.fixture
    def annotation(self):
        return Annotation(id=1, group=1, charge=1, isotopologue_index=1)

    def test_add_single_annotation(self, annotation):
        graph = AnnotationGraph()
        graph.add_annotation(annotation)

        node = graph.get_node(annotation.group)

        assert node.group == annotation.group
        assert len(graph.list_node_groups()) == 1
        assert node.charge_votes[annotation.group] == 1
        assert node.index_votes[annotation.group] == 1

    def test_add_multiple_times_annotation_with_same_group_updates_same_node(self, annotation: Annotation):
        graph = AnnotationGraph()
        n_times_added = 5
        for _ in range(n_times_added):
            graph.add_annotation(annotation)

        node = graph.get_node(annotation.group)

        assert node.group == annotation.group
        assert len(graph.list_node_groups()) == 1
        assert node.charge_votes[annotation.charge] == n_times_added
        assert node.index_votes[annotation.isotopologue_index] == n_times_added

    def test_add_multiple_annotation_with_equal_group_and_different_charge_state_update_votes(
        self, annotation: Annotation
    ):
        annotation2 = annotation.model_copy()
        annotation2.charge = 2

        graph = AnnotationGraph()
        graph.add_annotation(annotation)
        graph.add_annotation(annotation2)

        assert len(graph.list_node_groups()) == 1

        node = graph.get_node(annotation.group)
        assert node.group == annotation.group
        # both annotations have different charge but equal isotopologue index
        assert node.charge_votes[annotation.charge] == 1
        assert node.charge_votes[annotation2.charge] == 1
        assert node.index_votes[annotation.isotopologue_index] == 2

    def test_add_annotation_with_different_group_creates_new_nodes(self, annotation: Annotation):
        annotation2 = annotation.model_copy()
        annotation2.group = 2
        annotation2.charge = 2

        graph = AnnotationGraph()
        graph.add_annotation(annotation)
        graph.add_annotation(annotation2)

        assert len(graph.list_node_groups()) == 2

        node1 = graph.get_node(annotation.group)
        assert node1.group == annotation.group
        assert node1.charge_votes[annotation.charge] == 1
        assert node1.index_votes[annotation.isotopologue_index] == 1

        node2 = graph.get_node(annotation2.group)
        assert node2.group == annotation2.group
        assert node2.charge_votes[annotation2.charge] == 1
        assert node2.index_votes[annotation2.isotopologue_index] == 1

    def test_get_node_missing_node_raises_error(self):
        graph = AnnotationGraph()
        with pytest.raises(MissingNodeError):
            graph.get_node(1000)

    def test_remove_node_ok(self, annotation: Annotation):
        graph = AnnotationGraph()

        # add node and check that it is retrievable
        graph.add_annotation(annotation)
        graph.get_node(annotation.group)

        graph.remove_node(annotation.group)
        with pytest.raises(MissingNodeError):
            graph.get_node(annotation.group)

    def test_count_votes(self, annotation: Annotation):
        # add multiple times an annotation with same group but different
        # charge state. `charge` field should be set to 2 after counting votes

        graph = AnnotationGraph()

        votes_with_charge_one = 2
        for _ in range(votes_with_charge_one):
            graph.add_annotation(annotation)

        expected_node_charge = 2
        annotation.charge = expected_node_charge
        votes_with_charge_two = 3

        for _ in range(votes_with_charge_two):
            graph.add_annotation(annotation)

        graph.count_votes()

        node = graph.get_node(annotation.group)
        assert node.charge == expected_node_charge


class TestAnnotationGraphEdgeFunctionality:
    @pytest.fixture
    def graph(self):
        g = AnnotationGraph()

        for group in range(3):
            ann = Annotation(group=group, charge=1, isotopologue_index=0)
            g.add_annotation(ann)
        return g

    def test_add_single_edge(self, graph: AnnotationGraph):
        group1, group2 = 0, 1
        graph.add_edge(group1, group2)

        node1_edges = graph.get_edges(group1)
        assert len(node1_edges) == 1
        edge1 = node1_edges[0]
        assert edge1.groups == (group1, group2)
        assert edge1.votes == 1

        node2_edges = graph.get_edges(group2)
        assert len(node2_edges) == 1
        edge2 = node2_edges[0]
        assert edge2.groups == (group2, group1)
        assert edge2.votes == 1

    def test_add_multiple_times_same_edge(self, graph: AnnotationGraph):
        group1, group2 = 0, 1
        add_count = 5
        for _ in range(add_count):
            graph.add_edge(group1, group2)

        node1_edges = graph.get_edges(group1)
        assert len(node1_edges) == 1
        edge = node1_edges[0]
        assert edge.groups == (group1, group2)
        assert edge.votes == add_count

        node2_edges = graph.get_edges(group2)
        assert len(node2_edges) == 1
        edge2 = node2_edges[0]
        assert edge2.groups == (group2, group1)
        assert edge2.votes == add_count

    def test_remove_edges_ok(self, graph: AnnotationGraph):
        group1, group2 = 0, 1
        graph.add_edge(group1, group2)

        assert len(graph.get_edges(group1)) == 1
        assert len(graph.get_edges(group2)) == 1

        graph.remove_edge(group1, group2)

        assert len(graph.get_edges(group1)) == 0
        assert len(graph.get_edges(group2)) == 0

    def test_remove_edges_non_existing_edge_raises_error(self, graph: AnnotationGraph):
        with pytest.raises(ValueError):
            graph.remove_edge(0, 1)

    def test_no_neighbors_returns_empty_list(self, graph: AnnotationGraph):
        neighbors = graph.get_neighbors(0)
        assert len(neighbors) == 0

    def test_get_neighbors_ok(self, graph: AnnotationGraph):
        neighbor_groups = [1, 2]
        node_group = 0
        for neighbor_group in neighbor_groups:
            graph.add_edge(node_group, neighbor_group)

        neighbors = graph.get_neighbors(node_group)

        assert len(neighbors) == len(neighbor_groups)
        assert neighbors[0] == graph.get_node(neighbor_groups[0])
        assert neighbors[1] == graph.get_node(neighbor_groups[1])

    def test_remove_isolated_in_graph_with_no_edges_delete_all_nodes(self, graph: AnnotationGraph):
        graph.remove_isolated()
        all_node_groups = graph.list_node_groups()
        assert not all_node_groups

    def test_remove_isolated_ok(self, graph: AnnotationGraph):
        graph.add_edge(0, 1)
        graph.remove_isolated()

        all_node_groups = graph.list_node_groups()
        assert 0 in all_node_groups
        assert 1 in all_node_groups
        assert 2 not in all_node_groups


class TestAddSampleAnnotations:
    def test_annotation_with_no_group_are_not_added_to_the_graph(self):
        graph = AnnotationGraph()
        annotations = [Annotation(isotopologue_index=0), Annotation(isotopologue_index=0, group=1)]
        consensus.add_sample_annotations(graph, annotations)

        assert -1 not in graph.list_node_groups()

    def test_annotation_with_no_isotopologue_index_are_not_added_to_the_graph(self):
        graph = AnnotationGraph()
        exclude_group = 0
        annotations = [Annotation(group=exclude_group), Annotation(isotopologue_index=0, group=1)]
        consensus.add_sample_annotations(graph, annotations)

        assert exclude_group not in graph.list_node_groups()

    def test_non_zero_isotopologue_indices_only_neighbor_is_their_mmi(self):
        graph = AnnotationGraph()
        annotations = [
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1),
            Annotation(group=2, isotopologue_label=0, isotopologue_index=2),
            Annotation(group=3, isotopologue_label=1, isotopologue_index=0),
            Annotation(group=4, isotopologue_label=2, isotopologue_index=1),
        ]
        consensus.add_sample_annotations(graph, annotations)

        # check isotopologue label == 0 mmi
        group0_neighbors = [x.group for x in graph.get_neighbors(0)]
        assert len(group0_neighbors) == 2
        assert 1 in group0_neighbors
        assert 2 in group0_neighbors

        # check isotopologue label == 0 non mmi
        group1_neighbors = [x.group for x in graph.get_neighbors(1)]
        assert group1_neighbors == [0]

        group2_neighbors = [x.group for x in graph.get_neighbors(2)]
        assert group2_neighbors == [0]

        # check isotopologue label == 0 mmi
        group3_neighbors = [x.group for x in graph.get_neighbors(3)]
        assert not group3_neighbors


class TestRemoveIncompatibleNeighbors:
    def test_isotopologue_neighbor_with_equal_index_is_removed(self):
        graph = AnnotationGraph()
        annotations = [
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample1"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample1"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample3"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample4"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample5"),
        ]
        for sample_annotations in consensus.group_annotations_by_sample(annotations).values():
            consensus.add_sample_annotations(graph, sample_annotations)
        graph.count_votes()

        mmi = graph.get_node(0)

        assert 1 in mmi.neighbors

        consensus.remove_invalid_mmi_neighbors(graph, mmi)

        assert 1 not in mmi.neighbors

    def test_isotopologue_neighbor_with_different_charge_is_removed(self):
        graph = AnnotationGraph()
        annotations = [
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample1"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample1"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=2, sample_id="sample3"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=2, sample_id="sample4"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=2, sample_id="sample5"),
        ]
        for sample_annotations in consensus.group_annotations_by_sample(annotations).values():
            consensus.add_sample_annotations(graph, sample_annotations)
        graph.count_votes()

        mmi = graph.get_node(0)

        assert 1 in mmi.neighbors

        consensus.remove_invalid_mmi_neighbors(graph, mmi)

        assert 1 not in mmi.neighbors

    def test_valid_neighbor_is_kept(self):
        graph = AnnotationGraph()
        annotations = [
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample1"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample1"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample2"),
        ]
        for sample_annotations in consensus.group_annotations_by_sample(annotations).values():
            consensus.add_sample_annotations(graph, sample_annotations)

        graph.count_votes()

        mmi = graph.get_node(0)

        assert 1 in mmi.neighbors

        consensus.remove_invalid_mmi_neighbors(graph, mmi)

        assert 1 in mmi.neighbors


class TestSolveMMIConflicts:
    def test_remove_isotopologue_with_least_votes(self):
        graph = AnnotationGraph()
        annotations = [
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample1"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample2"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample3"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample4"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample5"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample1"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample2"),
            Annotation(group=2, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample3"),
            Annotation(group=2, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample4"),
            Annotation(group=2, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample5"),
        ]
        for sample_annotations in consensus.group_annotations_by_sample(annotations).values():
            consensus.add_sample_annotations(graph, sample_annotations)
        graph.count_votes()

        mmi = graph.get_node(0)

        assert 1 in mmi.neighbors
        assert 2 in mmi.neighbors

        consensus.solve_mmi_conflict(graph, mmi)

        assert 1 not in mmi.neighbors
        assert 2 in mmi.neighbors

    def test_all_valid_isotopologue_are_kept(self):
        graph = AnnotationGraph()
        annotations = [
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample1"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample1"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample2"),
            Annotation(group=2, isotopologue_label=0, isotopologue_index=2, charge=1, sample_id="sample1"),
            Annotation(group=2, isotopologue_label=0, isotopologue_index=2, charge=1, sample_id="sample2"),
        ]
        for sample_annotations in consensus.group_annotations_by_sample(annotations).values():
            consensus.add_sample_annotations(graph, sample_annotations)
        graph.count_votes()

        mmi = graph.get_node(0)

        assert 1 in mmi.neighbors
        assert 2 in mmi.neighbors

        consensus.solve_mmi_conflict(graph, mmi)

        assert 1 in mmi.neighbors
        assert 2 in mmi.neighbors


class TestSolveNonMMIConflict:
    def test_only_most_voted_mmi_neighbor_is_kept(self):
        graph = AnnotationGraph()
        annotations = [
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample1"),
            Annotation(group=0, isotopologue_label=0, isotopologue_index=0, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample1"),
            Annotation(group=1, isotopologue_label=0, isotopologue_index=1, charge=1, sample_id="sample2"),
            Annotation(group=1, isotopologue_label=1, isotopologue_index=1, charge=1, sample_id="sample3"),
            Annotation(group=2, isotopologue_label=1, isotopologue_index=0, charge=1, sample_id="sample3"),
        ]
        for sample_annotations in consensus.group_annotations_by_sample(annotations).values():
            consensus.add_sample_annotations(graph, sample_annotations)
        graph.count_votes()

        m1 = graph.get_node(1)

        assert 0 in m1.neighbors
        assert 2 in m1.neighbors

        consensus.solve_mmi_conflict(graph, m1)

        assert 0 in m1.neighbors
        assert 2 not in m1.neighbors
