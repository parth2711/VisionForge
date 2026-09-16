"""
Tests for the Feature Extraction Engine.

Uses synthetic images — no external test data required.
"""

import numpy as np
import pytest

from src.feature_extraction import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor()


@pytest.fixture
def synthetic_gray():
    """256×256 grayscale gradient."""
    return np.tile(np.arange(256, dtype=np.uint8), (256, 1))


@pytest.fixture
def checkerboard():
    """256×256 checkerboard pattern (strong corners)."""
    block = 32
    img = np.zeros((256, 256), dtype=np.uint8)
    for r in range(0, 256, block * 2):
        for c in range(0, 256, block * 2):
            img[r:r + block, c:c + block] = 255
            img[r + block:r + 2 * block, c + block:c + 2 * block] = 255
    return img


@pytest.fixture
def line_image():
    """256×256 image with a horizontal and vertical line."""
    img = np.zeros((256, 256), dtype=np.uint8)
    img[128, :] = 255   # horizontal line
    img[:, 128] = 255   # vertical line
    return img


# ── Canny edge detection ────────────────────────────────────────────

class TestCanny:
    def test_output_is_binary(self, extractor, synthetic_gray):
        edges = extractor.canny_edges(synthetic_gray)
        unique = np.unique(edges)
        assert all(v in (0, 255) for v in unique)

    def test_output_shape(self, extractor, synthetic_gray):
        edges = extractor.canny_edges(synthetic_gray)
        assert edges.shape == synthetic_gray.shape

    def test_edge_ratio_recorded(self, extractor, synthetic_gray):
        extractor.canny_edges(synthetic_gray)
        assert "canny_edge_ratio" in extractor.metadata

    def test_auto_threshold(self, extractor, synthetic_gray):
        extractor.canny_edges(synthetic_gray, auto_threshold=True)
        assert extractor.metadata["canny_thresholds"]["low"] >= 0
        assert extractor.metadata["canny_thresholds"]["high"] <= 255


# ── LoG edge detection ──────────────────────────────────────────────

class TestLoG:
    def test_output_shape(self, extractor, synthetic_gray):
        log = extractor.log_edges(synthetic_gray, sigma=1.0)
        assert log.shape == synthetic_gray.shape

    def test_output_dtype(self, extractor, synthetic_gray):
        log = extractor.log_edges(synthetic_gray)
        assert log.dtype == np.uint8


# ── DoG edge detection ──────────────────────────────────────────────

class TestDoG:
    def test_output_shape(self, extractor, synthetic_gray):
        dog = extractor.dog_edges(synthetic_gray)
        assert dog.shape == synthetic_gray.shape

    def test_different_sigmas_different_results(self, extractor, synthetic_gray):
        dog1 = extractor.dog_edges(synthetic_gray, sigma1=1.0, sigma2=2.0)
        dog2 = extractor.dog_edges(synthetic_gray, sigma1=2.0, sigma2=4.0)
        assert not np.array_equal(dog1, dog2)


# ── Harris corner detection ─────────────────────────────────────────

class TestHarris:
    def test_detects_corners_on_checkerboard(self, extractor, checkerboard):
        result = extractor.harris_corners(checkerboard)
        assert result["count"] > 0

    def test_returns_corner_image(self, extractor, checkerboard):
        result = extractor.harris_corners(checkerboard)
        assert result["corner_image"].ndim == 3  # BGR visualisation
        assert result["corner_image"].shape[:2] == checkerboard.shape

    def test_response_map_shape(self, extractor, checkerboard):
        result = extractor.harris_corners(checkerboard)
        assert result["response_map"].shape == checkerboard.shape

    def test_metadata_recorded(self, extractor, checkerboard):
        extractor.harris_corners(checkerboard)
        assert "harris_corners_count" in extractor.metadata


# ── HOG features ────────────────────────────────────────────────────

class TestHOG:
    def test_descriptor_length(self, extractor, synthetic_gray):
        result = extractor.hog_features(synthetic_gray)
        assert result["descriptor_length"] > 0
        assert len(result["descriptor"]) == result["descriptor_length"]

    def test_visualization_shape(self, extractor, synthetic_gray):
        result = extractor.hog_features(synthetic_gray)
        assert result["visualization"].shape == synthetic_gray.shape

    def test_energy_positive(self, extractor, synthetic_gray):
        extractor.hog_features(synthetic_gray)
        assert extractor.metadata["hog_energy"] > 0


# ── Scale-space analysis ────────────────────────────────────────────

class TestScaleSpace:
    def test_correct_number_of_scales(self, extractor, synthetic_gray):
        result = extractor.scale_space_analysis(synthetic_gray, num_scales=3)
        assert len(result["scales"]) == 3

    def test_sigma_increases(self, extractor, synthetic_gray):
        result = extractor.scale_space_analysis(synthetic_gray, num_scales=4)
        sigmas = [s["sigma"] for s in result["scales"]]
        assert all(sigmas[i] < sigmas[i + 1] for i in range(len(sigmas) - 1))

    def test_visualization_created(self, extractor, synthetic_gray):
        result = extractor.scale_space_analysis(synthetic_gray)
        assert result["visualization"].ndim == 3


# ── Gabor filters ───────────────────────────────────────────────────

class TestGabor:
    def test_filter_count(self, extractor, synthetic_gray):
        result = extractor.gabor_filter_bank(
            synthetic_gray,
            frequencies=[0.1, 0.2],
            orientations=[0, np.pi / 2],
        )
        assert len(result["responses"]) == 4  # 2 freq × 2 orient

    def test_response_shape(self, extractor, synthetic_gray):
        result = extractor.gabor_filter_bank(synthetic_gray)
        for resp in result["responses"]:
            assert resp["response"].shape == synthetic_gray.shape


# ── Full extraction pipeline ────────────────────────────────────────

class TestFullExtraction:
    def test_all_features_present(self, extractor, synthetic_gray):
        result = extractor.run_full_extraction(synthetic_gray)
        expected = {"canny", "log", "dog", "harris", "hog",
                    "scale_space", "gabor", "metadata"}
        assert expected.issubset(result.keys())
