"""
Tests for the Segmentation Engine.

Uses synthetic images — no external test data required.
"""

import numpy as np
import pytest

from src.segmentation import Segmenter


@pytest.fixture
def segmenter():
    return Segmenter()


@pytest.fixture
def three_color_image():
    """100×100 image with three distinct colour blocks."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:33, :, :] = [255, 0, 0]     # Blue block
    img[33:66, :, :] = [0, 255, 0]   # Green block
    img[66:, :, :] = [0, 0, 255]     # Red block
    return img


@pytest.fixture
def uniform_gray():
    """100×100 uniform gray image (good for region growing)."""
    img = np.full((100, 100), 128, dtype=np.uint8)
    # Add a bright square in the centre
    img[30:70, 30:70] = 200
    return img


@pytest.fixture
def gradient_gray():
    """256×256 smooth gradient."""
    return np.tile(np.arange(256, dtype=np.uint8), (256, 1))


# ── K-Means segmentation ────────────────────────────────────────────

class TestKMeans:
    def test_correct_k(self, segmenter, three_color_image):
        result = segmenter.kmeans_segmentation(three_color_image, k=3)
        unique_labels = np.unique(result["labels"])
        assert len(unique_labels) == 3

    def test_output_shape(self, segmenter, three_color_image):
        result = segmenter.kmeans_segmentation(three_color_image, k=3)
        assert result["segmented"].shape == three_color_image.shape

    def test_centers_shape(self, segmenter, three_color_image):
        result = segmenter.kmeans_segmentation(three_color_image, k=3)
        assert result["centers"].shape == (3, 3)

    def test_inertia_decreases_with_more_k(self, segmenter):
        # Use a richer image so k=5 can find 5 distinct clusters
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        r2 = segmenter.kmeans_segmentation(img, k=2)
        r5 = segmenter.kmeans_segmentation(img, k=5)
        assert r5["inertia"] <= r2["inertia"]

    def test_metadata_recorded(self, segmenter, three_color_image):
        segmenter.kmeans_segmentation(three_color_image, k=3)
        assert "kmeans_time" in segmenter.metadata


# ── Mean-Shift segmentation ─────────────────────────────────────────

class TestMeanShift:
    def test_output_shape(self, segmenter, three_color_image):
        result = segmenter.meanshift_segmentation(three_color_image)
        assert result["segmented"].shape == three_color_image.shape

    def test_produces_segments(self, segmenter, three_color_image):
        result = segmenter.meanshift_segmentation(three_color_image)
        assert result["num_segments"] >= 1


# ── Region growing ──────────────────────────────────────────────────

class TestRegionGrowing:
    def test_grows_connected_region(self, segmenter, uniform_gray):
        result = segmenter.region_growing(uniform_gray,
                                          seed_point=(50, 50),
                                          threshold=10)
        assert result["region_size"] > 0
        # Seed is inside the bright square (30:70, 30:70)
        assert result["mask"][50, 50] == 255

    def test_mask_shape(self, segmenter, uniform_gray):
        result = segmenter.region_growing(uniform_gray)
        assert result["mask"].shape == uniform_gray.shape

    def test_default_seed_is_centre(self, segmenter, uniform_gray):
        result = segmenter.region_growing(uniform_gray)
        assert result["seed_point"] == (50, 50)

    def test_tight_threshold_small_region(self, segmenter):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[48:52, 48:52] = 100
        result = segmenter.region_growing(img, seed_point=(50, 50),
                                          threshold=5)
        # Region should be small (just the bright patch)
        assert result["region_size"] <= 20


# ── Edge-based segmentation ─────────────────────────────────────────

class TestEdgeBased:
    def test_produces_regions(self, segmenter, three_color_image):
        result = segmenter.edge_based_segmentation(three_color_image)
        assert result["num_regions"] >= 0  # At least runs without error

    def test_edge_map_returned(self, segmenter, three_color_image):
        result = segmenter.edge_based_segmentation(three_color_image)
        assert result["edges"].shape == three_color_image.shape[:2]


# ── Watershed segmentation ──────────────────────────────────────────

class TestWatershed:
    def test_output_shape(self, segmenter, three_color_image):
        result = segmenter.watershed_segmentation(three_color_image)
        assert result["segmented"].shape == three_color_image.shape

    def test_produces_segments(self, segmenter, three_color_image):
        result = segmenter.watershed_segmentation(three_color_image)
        assert result["num_segments"] >= 1


# ── Method dispatch ─────────────────────────────────────────────────

class TestDispatch:
    def test_unknown_method_raises(self, segmenter, three_color_image):
        with pytest.raises(ValueError):
            segmenter.run_full_segmentation(three_color_image,
                                            method="nonexistent")

    def test_kmeans_via_dispatch(self, segmenter, three_color_image):
        result = segmenter.run_full_segmentation(three_color_image,
                                                 method="kmeans", k=3)
        assert result["method"] == "kmeans"
        assert "segmented" in result
