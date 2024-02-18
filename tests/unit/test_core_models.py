from __future__ import annotations

from typing import cast

import pytest
from tidyms.core import models

from .utils import ConcreteFeature, ConcreteRoi


@pytest.fixture
def expected_descriptor_names():
    return ["area", "height", "custom_descriptor", "mz"]


class TestRoi:
    def test_id(self):
        expected_id = 100
        roi = ConcreteRoi(id=expected_id)
        actual_id = roi.id
        assert actual_id == expected_id


    def test_serialization(self):
        expected = ConcreteRoi(data=[1.0, 2.0], id=10)
        serialized = expected.to_str()
        actual = cast(ConcreteRoi, ConcreteRoi.from_str(serialized))

        assert actual.data == expected.data
        assert actual.id == expected.id


    def test_add_feature(self):
        roi = ConcreteRoi()
        ft = ConcreteFeature(roi=roi, data=1)
        roi.add_feature(ft)
        assert ft in roi.features

    def test_add_multiple_features(self):
        roi = ConcreteRoi()
        ft1 = ConcreteFeature(roi=roi, data=1)
        roi.add_feature(ft1)
        ft2 = ConcreteFeature(roi=roi, data=1)
        roi.add_feature(ft2)
        assert ft1 in roi.features
        assert ft2 in roi.features
        assert ft1 is not ft2


    def test_remove_feature(self):
        roi = ConcreteRoi()
        ft1 = ConcreteFeature(roi=roi, data=1)
        roi.add_feature(ft1)
        assert ft1 in roi.features
        roi.remove_feature(ft1)
        assert ft1 not in roi.features


class TestFeature:

    @pytest.fixture(scope="class")
    def roi(self) -> ConcreteRoi:
        return ConcreteRoi(data=[1.0, 2.0], id=1)

    @pytest.fixture(scope="class")
    def feature(self, roi) -> ConcreteFeature:
        return ConcreteFeature(data=1, roi=roi)

    def test_serialization(self, feature):
        serialized = feature.to_str()
        expected = cast(
            ConcreteFeature,
            ConcreteFeature.from_str(serialized, feature.roi, feature.annotation),
        )
        assert expected.data == feature.data
        assert expected.roi == feature.roi
        assert expected.id == feature.id


    def test_mz_equals_get_mz(self, feature):
        assert feature.get("mz") == feature.mz


    def test_area_equals_get_area(self, feature):
        assert feature.get("area") == feature.area


    def test_height_equals_get_height(self, feature):
        assert feature.get("height") == feature.height


    def test_custom_descriptor_equals_get_custom_descriptor(self, feature):
        assert feature.get("custom_descriptor") == feature.custom_descriptor


    def test_descriptor_names_are_feature_attributes(self, feature):
        all_descriptors = feature.descriptor_names()
        all_attr = feature.__dict__
        assert all(x in all_attr for x in all_descriptors)


    def test_describe(self, feature):
        descriptors = feature.describe()
        all_attr = feature.__dict__
        assert all(x in all_attr for x in descriptors)
        assert all(isinstance(x, float) for x in descriptors.values())


    def test_order_lt(self, roi):
        ft1 = ConcreteFeature(roi=roi, data=1)
        ft2 = ConcreteFeature(roi=roi, data=2)
        assert ft1 < ft2


    def test_le(self, roi):
        ft1 = ConcreteFeature(roi=roi, data=1)
        ft2 = ConcreteFeature(roi=roi, data=1)
        assert ft1 <= ft2


    def test_order_ge(self, roi):
        ft1 = ConcreteFeature(roi=roi, data=1)
        ft2 = ConcreteFeature(roi=roi, data=1)
        assert ft1 >= ft2


    def test_order_gt(self, roi):
        ft1 = ConcreteFeature(roi=roi, data=2)
        ft2 = ConcreteFeature(roi=roi, data=1)
        assert ft1 >= ft2


class TestSample:
    def test_serialization(self, tmp_path):
        sample_id = "my-sample"
        path = tmp_path / sample_id
        expected = models.Sample(id=sample_id, path=path, batch=2)
        actual = models.Sample(**expected.model_dump())
        assert actual == expected

    def test_serialization_with_extra(self, tmp_path):
        sample_id = "my-sample"
        path = tmp_path / sample_id
        extra = {"extra-field-1": 0.25, "extra-field-2": 3, "extra-field-3": "extra"}
        expected = models.Sample(id=sample_id, path=path, batch=2, extra=extra)
        actual = models.Sample(**expected.model_dump())
        assert actual == expected


class TestSampleData:

    @pytest.fixture
    def sample(self, tmp_path):
        id_ = "my-sample"
        return models.Sample(id=id_, path=tmp_path / id_)

    def test_get_features_no_roi(self, sample):
        sample_data = models.SampleData(sample=sample)
        expected = list()
        actual = sample_data.get_features()
        assert actual == expected

    def test_get_features_with_roi_no_features(self, sample):
        rois = [ConcreteRoi() for _ in range(5)]
        sample_data = models.SampleData(sample=sample, roi=rois)
        expected = list()
        actual = sample_data.get_features()
        assert actual == expected

    def test_get_features_with_roi_and_features(self, sample):
        rois = [ConcreteRoi() for _ in range(5)]
        n_ft = 4
        roi = rois[2]
        features = [ConcreteFeature(roi=roi, data=1) for _ in range(n_ft)]
        for ft in features:
            roi.add_feature(ft)
        sample_data = models.SampleData(sample=sample, roi=rois)
        actual = sample_data.get_features()
        assert len(actual) == n_ft
