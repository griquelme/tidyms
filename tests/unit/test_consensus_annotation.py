from tidyms import consensus_annotation
from tidyms import _constants as c
import pandas as pd
import pytest
from collections import Counter

@pytest.fixture
def feature_table():
    # Three feature labels, all belonging to the same envelope
    # rows with -1 are noise.
    columns = [c.SAMPLE, c.LABEL, c.ENVELOPE_LABEL, c.ENVELOPE_INDEX, c.CHARGE]
    data = [
        [0, -1, -1, -1, -1],
        [1, -1, -1, -1, -1],
        [2, -1, -1, -1, -1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [2, 0, 0, 0, 1],
        [3, 0, 0, 1, 1],
        [4, 0, 0, 0, 2],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [2, 1, 0, 1, 1],
        [3, 1, 0, 2, 1],
        [4, 1, 0, 1, 2],
        [0, 2, 0, 2, 1],
        [1, 2, 0, 2, 1],
        [2, 2, 0, 2, 1],
        [3, 2, 0, 2, 1],
        [4, 2, 0, 2, 2],
    ]
    return pd.DataFrame(data=data, columns=columns)


def test__build_graph(feature_table):
    graph, annotations = consensus_annotation.vote_annotations(feature_table)
    assert len(annotations) == 3
    for ft_label, ft_data in annotations.items():
        assert ft_data[c.CHARGE] == 1
        assert ft_data[c.ENVELOPE_LABEL] == 0
        assert ft_data[c.ENVELOPE_INDEX] == ft_label


def test__build_graph_nodes(feature_table):
    nodes = consensus_annotation._build_graph_nodes(feature_table)
    expected = {
        0: {c.CHARGE: 1, c.ENVELOPE_INDEX: 0},
        1: {c.CHARGE: 1, c.ENVELOPE_INDEX: 1},
        2: {c.CHARGE: 1, c.ENVELOPE_INDEX: 2}
    }
    assert nodes == expected

def test__build_graph_edges(feature_table):
    edges = consensus_annotation._build_graph_edges(feature_table)
    edge_count = Counter(edges)
    expected = Counter({(0, 1): 4, (0, 2): 4})
    assert edge_count == expected
