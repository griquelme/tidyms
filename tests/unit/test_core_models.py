from __future__ import annotations

from typing import cast

import pytest

from .utils import ConcreteFeature, ConcreteRoi


@pytest.fixture
def roi():
    return ConcreteRoi(data=[1.0, 2.0])


@pytest.fixture
def feature(roi: ConcreteRoi):
    return ConcreteFeature(roi=roi, data=1)


@pytest.fixture
def expected_descriptor_names():
    return ["area", "height", "custom_descriptor", "mz"]


def test_concrete_roi_id():
    expected_id = 100
    roi = ConcreteRoi(id=expected_id)
    actual_id = roi.id
    assert actual_id == expected_id


def test_concrete_roi_serialization(roi: ConcreteRoi):
    expected = roi
    serialized = expected.to_str()
    actual = cast(ConcreteRoi, ConcreteRoi.from_str(serialized))

    assert actual.data == expected.data
    assert actual.id == expected.id


def test_Roi_add_feature():
    roi = ConcreteRoi()
    ft = ConcreteFeature(roi=roi, data=1)
    roi.add_feature(ft)
    assert ft in roi.features


def test_Roi_add_feature_add_multiple_features():
    roi = ConcreteRoi()
    ft1 = ConcreteFeature(roi=roi, data=1)
    roi.add_feature(ft1)
    ft2 = ConcreteFeature(roi=roi, data=1)
    roi.add_feature(ft2)
    assert ft1 in roi.features
    assert ft2 in roi.features
    assert ft1 is not ft2


def test_Roi_remove_feature():
    roi = ConcreteRoi()
    ft1 = ConcreteFeature(roi=roi, data=1)
    roi.add_feature(ft1)
    assert ft1 in roi.features
    roi.remove_feature(ft1)
    assert ft1 not in roi.features


def test_concrete_feature_serialization(feature: ConcreteFeature):
    serialized = feature.to_str()
    expected = cast(
        ConcreteFeature,
        ConcreteFeature.from_str(serialized, feature.roi, feature.annotation),
    )
    assert expected.data == feature.data
    assert expected.roi == feature.roi
    assert expected.id == feature.id


def test_feature_mz_equals_get_mz(feature: ConcreteFeature):
    feature._set_mz()
    assert feature.mz == feature.get("mz")


def test_feature_area_equals_get_area(feature: ConcreteFeature):
    feature._set_area()
    assert feature.area == feature.get("area")


def test_feature_height_equals_get_height(feature: ConcreteFeature):
    feature._set_height()
    assert feature.height == feature.get("height")


def test_Feature_get_custom_descriptor(feature: ConcreteFeature):
    feature._set_custom_descriptor()
    assert feature.custom_descriptor == feature.get("custom_descriptor")


def test_feature_descriptor_names(expected_descriptor_names: list[str]):
    all_descriptors = ConcreteFeature.descriptor_names()
    assert all(d in all_descriptors for d in expected_descriptor_names)
    assert len(all_descriptors) == 4


def test_Feature_describe(
    expected_descriptor_names: list[str], feature: ConcreteFeature
):
    descriptors = feature.describe()
    assert all(d in descriptors for d in expected_descriptor_names)
    assert all(isinstance(x, float) for x in descriptors.values())


def test_Feature_order_lt(roi: ConcreteRoi):
    ft1 = ConcreteFeature(roi=roi, data=1)
    ft2 = ConcreteFeature(roi=roi, data=2)
    assert ft1 < ft2


def test_Feature_order_le(roi: ConcreteRoi):
    ft1 = ConcreteFeature(roi=roi, data=1)
    ft2 = ConcreteFeature(roi=roi, data=1)
    assert ft1 <= ft2


def test_Feature_order_ge(roi: ConcreteRoi):
    ft1 = ConcreteFeature(roi=roi, data=1)
    ft2 = ConcreteFeature(roi=roi, data=1)
    assert ft1 >= ft2


def test_Feature_order_gt(roi: ConcreteRoi):
    ft1 = ConcreteFeature(roi=roi, data=2)
    ft2 = ConcreteFeature(roi=roi, data=1)
    assert ft1 >= ft2
