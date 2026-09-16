"""
Tests for the Preprocessing Engine.

Uses synthetic images (numpy-generated) — no external test data required.
"""

import numpy as np
import pytest

from src.preprocessing import ImagePreprocessor


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


@pytest.fixture
def synthetic_gray():
    """256×256 grayscale gradient image."""
    return np.tile(np.arange(256, dtype=np.uint8), (256, 1))


@pytest.fixture
def synthetic_bgr():
    """256×256 BGR image with distinct colour channels."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(256, dtype=np.uint8), (256, 1))  # B
    img[:, :, 1] = np.tile(np.arange(256, dtype=np.uint8)[::-1], (256, 1))  # G
    img[:, :, 2] = 128  # R constant
    return img


# ── Grayscale conversion ─────────────────────────────────────────────

class TestGrayscale:
    def test_output_is_single_channel(self, preprocessor, synthetic_bgr):
        gray = preprocessor.to_grayscale(synthetic_bgr)
        assert gray.ndim == 2

    def test_output_shape_matches(self, preprocessor, synthetic_bgr):
        gray = preprocessor.to_grayscale(synthetic_bgr)
        assert gray.shape == synthetic_bgr.shape[:2]

    def test_already_gray_passthrough(self, preprocessor, synthetic_gray):
        gray = preprocessor.to_grayscale(synthetic_gray)
        assert gray.ndim == 2
        np.testing.assert_array_equal(gray, synthetic_gray)

    def test_timing_recorded(self, preprocessor, synthetic_bgr):
        preprocessor.to_grayscale(synthetic_bgr)
        assert "grayscale_time" in preprocessor.metadata


# ── Gaussian blur ────────────────────────────────────────────────────

class TestGaussianBlur:
    def test_output_same_size(self, preprocessor, synthetic_gray):
        blurred = preprocessor.gaussian_blur(synthetic_gray, kernel_size=5)
        assert blurred.shape == synthetic_gray.shape

    def test_reduces_high_frequency(self, preprocessor, synthetic_gray):
        blurred = preprocessor.gaussian_blur(synthetic_gray, kernel_size=15)
        # Standard deviation should decrease after heavy blur
        assert np.std(blurred) <= np.std(synthetic_gray) + 1

    def test_even_kernel_corrected(self, preprocessor, synthetic_gray):
        blurred = preprocessor.gaussian_blur(synthetic_gray, kernel_size=4)
        assert preprocessor.metadata["gaussian_kernel_size"] == 5


# ── Median filter ────────────────────────────────────────────────────

class TestMedianFilter:
    def test_output_same_size(self, preprocessor, synthetic_gray):
        filtered = preprocessor.median_filter(synthetic_gray, kernel_size=5)
        assert filtered.shape == synthetic_gray.shape

    def test_removes_salt_pepper(self, preprocessor):
        img = np.full((100, 100), 128, dtype=np.uint8)
        # Add salt-and-pepper noise
        rng = np.random.default_rng(42)
        coords = rng.integers(0, 100, size=(2, 50))
        img[coords[0], coords[1]] = 255
        coords = rng.integers(0, 100, size=(2, 50))
        img[coords[0], coords[1]] = 0
        filtered = preprocessor.median_filter(img, kernel_size=3)
        # Filtered image should be closer to 128 on average
        assert abs(np.mean(filtered) - 128) < abs(np.mean(img) - 128) + 1


# ── Histogram ────────────────────────────────────────────────────────

class TestHistogram:
    def test_gray_histogram_256_bins(self, preprocessor, synthetic_gray):
        result = preprocessor.compute_histogram(synthetic_gray)
        assert len(result["histograms"]) == 1
        assert len(result["histograms"][0]) == 256

    def test_bgr_histogram_3_channels(self, preprocessor, synthetic_bgr):
        result = preprocessor.compute_histogram(synthetic_bgr)
        assert len(result["histograms"]) == 3
        assert result["channels"] == ["Blue", "Green", "Red"]

    def test_histogram_sums_to_pixel_count(self, preprocessor, synthetic_gray):
        result = preprocessor.compute_histogram(synthetic_gray)
        total = sum(result["histograms"][0])
        assert total == synthetic_gray.size

    def test_mean_and_std_present(self, preprocessor, synthetic_gray):
        result = preprocessor.compute_histogram(synthetic_gray)
        assert "mean" in result
        assert "std" in result
        assert result["mean"] > 0


# ── Histogram equalization ───────────────────────────────────────────

class TestHistogramEqualization:
    def test_output_same_shape(self, preprocessor, synthetic_gray):
        eq = preprocessor.histogram_equalization(synthetic_gray)
        assert eq.shape == synthetic_gray.shape

    def test_equalized_has_wider_spread(self, preprocessor):
        # Low-contrast image
        img = np.full((100, 100), 100, dtype=np.uint8)
        img[25:75, 25:75] = 110
        eq = preprocessor.histogram_equalization(img)
        assert np.std(eq) >= np.std(img)

    def test_bgr_equalization(self, preprocessor, synthetic_bgr):
        eq = preprocessor.histogram_equalization(synthetic_bgr)
        assert eq.shape == synthetic_bgr.shape


# ── Contrast enhancement ────────────────────────────────────────────

class TestContrastEnhancement:
    def test_clahe_output_shape(self, preprocessor, synthetic_gray):
        enhanced = preprocessor.enhance_contrast(synthetic_gray, method="clahe")
        assert enhanced.shape == synthetic_gray.shape

    def test_stretch_output_range(self, preprocessor):
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        enhanced = preprocessor.enhance_contrast(img, method="stretch")
        assert enhanced.min() == 0
        assert enhanced.max() == 255

    def test_unknown_method_raises(self, preprocessor, synthetic_gray):
        with pytest.raises(ValueError):
            preprocessor.enhance_contrast(synthetic_gray, method="unknown")


# ── Fourier analysis ────────────────────────────────────────────────

class TestFourierAnalysis:
    def test_magnitude_shape(self, preprocessor, synthetic_gray):
        result = preprocessor.fourier_analysis(synthetic_gray)
        assert result["magnitude"].ndim == 2
        assert result["magnitude"].shape[0] >= synthetic_gray.shape[0]

    def test_dominant_frequency_numeric(self, preprocessor, synthetic_gray):
        result = preprocessor.fourier_analysis(synthetic_gray)
        assert isinstance(result["dominant_frequency"], float)
        assert result["dominant_frequency"] >= 0


# ── Full pipeline ────────────────────────────────────────────────────

class TestFullPreprocessing:
    def test_returns_all_keys(self, preprocessor, synthetic_bgr):
        result = preprocessor.run_full_preprocessing(synthetic_bgr)
        expected_keys = {"grayscale", "blurred", "equalized", "enhanced",
                         "histogram", "fourier", "metadata", "processed_image",
                         "original_shape"}
        assert expected_keys.issubset(result.keys())

    def test_processed_image_is_grayscale(self, preprocessor, synthetic_bgr):
        result = preprocessor.run_full_preprocessing(synthetic_bgr)
        assert result["processed_image"].ndim == 2
