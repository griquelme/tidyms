from collections.abc import Sequence

import pytest
from tidyms.annotation import envelope_finder as ef
from tidyms.annotation.annotation_data import AnnotationData
from tidyms.chem import Formula, PeriodicTable

from .utils import ConcreteFeature, ConcreteRoi

# Variable name convention
# m denote nominal mass values
# M denote exact mass values
# p denote abundance


@pytest.fixture
def formula_groups():
    formulas = {
        "cho": ["C27H34O9", "C62H120O6", "C59H114O6", "C62H120O6", "C56H42O10"],
        "chnops": ["C41H80NO8P", "C54H104O6", "C27H40O2", "C24H26O12", "C55H106O6"],
    }
    return formulas


def create_feature_list_from_formula(f_str: str) -> Sequence[ConcreteFeature]:
    formula = Formula(f_str)
    M, _ = formula.get_isotopic_envelope()
    mz_list = M / abs(formula.charge) if formula.charge else M
    features = [ConcreteFeature(data=0, mz=mz, roi=ConcreteRoi(), id=k) for k, mz in enumerate(mz_list)]
    return features


class TestEnvelopeFinderComponents:

    element_groups = {"CHO": ["C", "H", "O"], "CHNOPS": ["C", "H", "N", "O", "P", "S"]}
    formula_groups = {
        "CHO": ["C27H34O9", "C62H120O6", "C59H114O6", "C62H120O6", "C56H42O10"],
        "CHNOPS": ["C41H80NO8P", "C54H104O6", "C27H40O2", "C24H26O12", "C55H106O6"],
    }
    max_length = 10
    mz_tol = 0.005
    min_similarity = 0.9

    @pytest.mark.parametrize("group", ["CHO", "CHNOPS"])
    def test__make_exact_mass_difference_bounds(self, group):
        # test bounds for different element combinations
        elements = [PeriodicTable().get_element(x) for x in self.element_groups[group]]
        bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        # m and M are the bounds for each nominal mass increment
        for e in elements:
            m, M, _ = e.get_abundances()
            m = m - m[0]
            M = M - M[0]
            for i, mi in zip(m[1:], M[1:]):
                m_min, m_max = bounds[i]
                assert m_min <= mi
                assert m_max >= mi

    @pytest.mark.parametrize("group", ["CHO", "CHNOPS"])
    def test__get_next_mz_search_interval_mz(self, group):
        elements = [PeriodicTable().get_element(x) for x in self.element_groups[group]]
        dM_bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        # test bounds for different formulas
        for f_str in self.formula_groups[group]:
            feature_list = create_feature_list_from_formula(f_str)
            length = len(feature_list)
            for k in range(1, length - 1):
                k_ft = feature_list[k]
                min_mz, max_mz = ef._get_next_mz_search_interval(feature_list[:k], dM_bounds, 1, 0.005)
                assert (min_mz < k_ft.mz) and (k_ft.mz < max_mz)

    @pytest.mark.parametrize("charge", list(range(1, 6)))
    def test_get_k_bounds_multiple_charges(self, charge):
        formulas = self.formula_groups["CHNOPS"]
        elements = [PeriodicTable().get_element(x) for x in self.element_groups["CHNOPS"]]
        bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        for f_str in formulas:
            features = create_feature_list_from_formula(f"[{f_str}]{charge}+")
            length = len(features)
            for k in range(1, length - 1):
                m_min, m_max = ef._get_next_mz_search_interval(features[:k], bounds, charge, 0.005)
                assert (m_min < features[k]) and (features[k] < m_max)


    @pytest.mark.parametrize("group,charge", [["CHO", 1], ["CHO", 2], ["CHNOPS", 1], ["CHNOPS", 2]])
    def test__find_envelopes(self, group, charge):
        # test that the function works using as a list m/z values generated from
        # formulas.
        formulas = self.formula_groups[group]
        elements = [PeriodicTable().get_element(x) for x in self.element_groups[group]]
        bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        for f_str in formulas:
            features = create_feature_list_from_formula(f"[{f_str}]{charge}+")
            data = AnnotationData(features)
            mmi = data.features[0]
            results = ef._find_envelopes(
                data.features,
                mmi,
                data.non_annotated,
                data.similarity_cache,
                charge,
                self.max_length,
                self.mz_tol,
                self.min_similarity,
                bounds,
            )
            expected = features
            assert results[0] == expected


    @pytest.mark.parametrize("group", ["CHO", "CHNOPS"])
    def test__find_envelopes_no_charge(self, group):
        # test that the function works using as a list m/z values generated from
        # formulas.
        elements = [PeriodicTable().get_element(x) for x in self.element_groups[group]]
        bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        charge = 0
        for f_str in self.formula_groups[group]:
            features = create_feature_list_from_formula(f_str)
            data = AnnotationData(features)
            mmi = features[0]
            results = ef._find_envelopes(
                features,
                mmi,
                data.non_annotated,
                data.similarity_cache,
                charge,
                self.max_length,
                self.mz_tol,
                self.min_similarity,
                bounds,
            )
            expected = features
            assert results[0] == expected


    @pytest.mark.parametrize("group", ["CHNOPS", "CHO"])
    def test_EnvelopeFinder(self, group):
        elements = self.element_groups[group]
        formulas = self.formula_groups[group]
        envelope_finder = ef.EnvelopeFinder(elements, mz_tolerance=self.mz_tol, max_length=self.max_length)
        charge = 1
        for f_str in formulas:
            features = create_feature_list_from_formula(f_str)
            mmi = features[0]
            data = AnnotationData(features)
            results = envelope_finder.find(data, mmi, charge)
            expected = features
            assert len(results) == 1
            assert results[0] == expected
