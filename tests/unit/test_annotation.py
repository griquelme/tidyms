# import numpy as np
# from tidyms import annotation
# from tidyms.lcms import Peak, LCRoi
# from math import isclose


# def test__overlap_ratio_overlapping_peaks():
#     scans = np.arange(100)
#     scans_roi1 = scans[20:40]
#     scans_roi2 = scans[25:55]
#     roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
#     roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
#     ft1 = Peak(5, 10, 15)
#     ft2 = Peak(5, 10, 15)
#     test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
#     expected_result = 0.5
#     assert isclose(expected_result, test_result)
#
#
# def test__overlap_ratio_non_overlapping_peaks():
#     scans = np.arange(100)
#     scans_roi1 = scans[20:40]
#     scans_roi2 = scans[30:50]
#     roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
#     roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
#     ft1 = Peak(5, 10, 15)
#     ft2 = Peak(15, 16, 20)
#     test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
#     expected_result = 0.0
#     assert isclose(expected_result, test_result)
#
#
# def test__overlap_ratio_perfect_overlap():
#     scans = np.arange(100)
#     scans_roi1 = scans[20:40]
#     scans_roi2 = scans[30:50]
#     roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
#     roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
#     ft1 = Peak(10, 15, 20)
#     ft2 = Peak(0, 5, 10)
#     test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
#     expected_result = 1.0
#     assert isclose(expected_result, test_result)
#
#
# def test__overlap_ratio_ft2_contained_in_ft1():
#     scans = np.arange(100)
#     scans_roi1 = scans[20:40]
#     scans_roi2 = scans[30:50]
#     roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
#     roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
#     ft1 = Peak(10, 15, 20)
#     ft2 = Peak(2, 5, 8)
#     test_result = annotation._overlap_ratio(roi1, ft1, roi2, ft2)
#     expected_result = 0.6   # ft2 extension / ft1 extension
#     assert isclose(expected_result, test_result)
#
#
# def test__feature_similarity_same_features():
#     scans = np.arange(100)
#     scans_roi1 = scans[20:40]
#     roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
#     ft1 = Peak(10, 15, 20)
#     min_overlap = 0.5
#     test_result = annotation._feature_similarity(
#         roi1, ft1, roi1, ft1, min_overlap)
#     expected_result = 1.0
#     assert isclose(expected_result, test_result)
#
#
# def test__feature_similarity_non_overlapping_features():
#     scans = np.arange(100)
#     scans_roi1 = scans[20:40]
#     roi1 = LCRoi(scans_roi1, scans_roi1, scans_roi1, scans_roi1)
#     ft1 = Peak(10, 15, 20)
#     scans_roi2 = scans[50:70]
#     roi2 = LCRoi(scans_roi2, scans_roi2, scans_roi2, scans_roi2)
#     ft2 = Peak(10, 15, 20)
#     min_overlap = 0.5
#     test_result = annotation._feature_similarity(
#         roi1, ft1, roi2, ft2, min_overlap)
#     expected_result = 0.0
#     assert isclose(expected_result, test_result)