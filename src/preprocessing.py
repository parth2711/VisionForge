"""
Preprocessing Engine — Module 1: Digital Image Formation & Low-Level Processing.

Implements grayscale conversion, noise reduction, histogram analysis,
contrast enhancement, and Fourier analysis.
"""

import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """
    Preprocessing engine for images.

    Provides methods for grayscale conversion, noise reduction (Gaussian blur,
    median filter), histogram computation, histogram equalization, CLAHE
    contrast enhancement, and Fourier-domain analysis.

    Each method records timing metadata so the evaluation engine can
    report per-operation performance.
    """

    def __init__(self):
        self.metadata = {}

    # ------------------------------------------------------------------
    # Colour-space conversion
    # ------------------------------------------------------------------

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR image to single-channel grayscale.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image (H×W×3).

        Returns
        -------
        np.ndarray
            Grayscale image (H×W).
        """
        start = time.time()
        if len(image.shape) == 2:
            gray = image.copy()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.metadata["grayscale_time"] = time.time() - start
        return gray

    # ------------------------------------------------------------------
    # Noise reduction
    # ------------------------------------------------------------------

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur for noise reduction.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or BGR).
        kernel_size : int
            Size of the Gaussian kernel (must be odd).

        Returns
        -------
        np.ndarray
            Blurred image.
        """
        start = time.time()
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        self.metadata["gaussian_blur_time"] = time.time() - start
        self.metadata["gaussian_kernel_size"] = kernel_size
        return blurred

    def median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filter for salt-and-pepper noise removal.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        kernel_size : int
            Aperture size (must be odd).

        Returns
        -------
        np.ndarray
            Filtered image.
        """
        start = time.time()
        if kernel_size % 2 == 0:
            kernel_size += 1
        filtered = cv2.medianBlur(image, kernel_size)
        self.metadata["median_filter_time"] = time.time() - start
        self.metadata["median_kernel_size"] = kernel_size
        return filtered

    # ------------------------------------------------------------------
    # Histogram processing
    # ------------------------------------------------------------------

    def compute_histogram(self, image: np.ndarray, output_path: str = None) -> dict:
        """Compute intensity histogram(s) and optionally save a plot.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or BGR).
        output_path : str, optional
            If provided, save the histogram plot to this path.

        Returns
        -------
        dict
            ``{"histograms": list[np.ndarray], "channels": list[str],
               "mean": float, "std": float, "plot_path": str | None}``
        """
        start = time.time()
        result = {"plot_path": None}

        if len(image.shape) == 2:
            # Grayscale
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            result["histograms"] = [hist.tolist()]
            result["channels"] = ["gray"]
            result["mean"] = float(np.mean(image))
            result["std"] = float(np.std(image))
        else:
            # BGR → compute per-channel
            colors = ("b", "g", "r")
            labels = ("Blue", "Green", "Red")
            hists = []
            for i, (col, lbl) in enumerate(zip(colors, labels)):
                h = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
                hists.append(h.tolist())
            result["histograms"] = hists
            result["channels"] = list(labels)
            result["mean"] = float(np.mean(image))
            result["std"] = float(np.std(image))

        if output_path:
            fig, ax = plt.subplots(figsize=(8, 4))
            if len(image.shape) == 2:
                ax.plot(result["histograms"][0], color="black", linewidth=0.8)
                ax.fill_between(range(256), result["histograms"][0],
                                alpha=0.3, color="gray")
            else:
                for h, col, lbl in zip(result["histograms"], ("blue", "green", "red"),
                                       result["channels"]):
                    ax.plot(h, color=col, linewidth=0.8, label=lbl)
                ax.legend()
            ax.set_title("Intensity Histogram")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            ax.set_xlim([0, 255])
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            result["plot_path"] = output_path

        self.metadata["histogram_time"] = time.time() - start
        return result

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to improve contrast.

        For colour images the equalization is applied to the V channel
        in HSV colour-space.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Contrast-enhanced image.
        """
        start = time.time()
        if len(image.shape) == 2:
            equalized = cv2.equalizeHist(image)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.metadata["hist_eq_time"] = time.time() - start
        return equalized

    # ------------------------------------------------------------------
    # Contrast enhancement
    # ------------------------------------------------------------------

    def enhance_contrast(self, image: np.ndarray, method: str = "clahe",
                         clip_limit: float = 2.0,
                         tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """Enhance image contrast using CLAHE or simple stretching.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        method : str
            ``'clahe'`` for Contrast-Limited Adaptive Histogram Equalization,
            ``'stretch'`` for min–max stretching.
        clip_limit : float
            CLAHE clip limit (only used when *method* is ``'clahe'``).
        tile_grid_size : tuple
            CLAHE tile grid size.

        Returns
        -------
        np.ndarray
            Enhanced image.
        """
        start = time.time()
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=tile_grid_size)
            if len(image.shape) == 2:
                enhanced = clahe.apply(image)
            else:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif method == "stretch":
            if len(image.shape) == 2:
                min_val, max_val = image.min(), image.max()
                if max_val - min_val == 0:
                    enhanced = image.copy()
                else:
                    enhanced = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                enhanced = np.zeros_like(image)
                for c in range(3):
                    ch = image[:, :, c]
                    mn, mx = ch.min(), ch.max()
                    if mx - mn == 0:
                        enhanced[:, :, c] = ch
                    else:
                        enhanced[:, :, c] = ((ch - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")

        self.metadata["enhancement_time"] = time.time() - start
        self.metadata["enhancement_method"] = method
        return enhanced

    # ------------------------------------------------------------------
    # Fourier analysis
    # ------------------------------------------------------------------

    def fourier_analysis(self, image: np.ndarray,
                         output_path: str = None) -> dict:
        """Compute the 2-D DFT magnitude and phase spectra.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale image.
        output_path : str, optional
            If provided, save the magnitude spectrum visualisation.

        Returns
        -------
        dict
            ``{"magnitude": np.ndarray, "phase": np.ndarray,
               "dominant_frequency": float, "plot_path": str | None}``
        """
        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Optimal DFT size
        rows, cols = image.shape
        opt_rows = cv2.getOptimalDFTSize(rows)
        opt_cols = cv2.getOptimalDFTSize(cols)
        padded = np.zeros((opt_rows, opt_cols), dtype=np.float32)
        padded[:rows, :cols] = image

        # DFT
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=[0, 1])

        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        phase = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Log-scale magnitude for visualisation
        magnitude_log = np.log1p(magnitude)
        magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)

        # Dominant frequency (distance of peak from centre)
        cy, cx = opt_rows // 2, opt_cols // 2
        mag_copy = magnitude_log.copy()
        # Mask the DC component
        mag_copy[cy - 5:cy + 5, cx - 5:cx + 5] = 0
        peak_loc = np.unravel_index(np.argmax(mag_copy), mag_copy.shape)
        dominant_freq = float(np.sqrt((peak_loc[0] - cy) ** 2 +
                                      (peak_loc[1] - cx) ** 2))

        result = {
            "magnitude": magnitude_norm,
            "phase": phase,
            "dominant_frequency": dominant_freq,
            "plot_path": None,
        }

        if output_path:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(magnitude_norm, cmap="gray")
            axes[0].set_title("Magnitude Spectrum (log scale)")
            axes[0].axis("off")
            axes[1].imshow(phase, cmap="hsv")
            axes[1].set_title("Phase Spectrum")
            axes[1].axis("off")
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            result["plot_path"] = output_path

        self.metadata["fourier_time"] = time.time() - start
        self.metadata["dominant_frequency"] = dominant_freq
        return result

    # ------------------------------------------------------------------
    # Full preprocessing pipeline
    # ------------------------------------------------------------------

    def run_full_preprocessing(self, image: np.ndarray,
                               output_dir: str = None) -> dict:
        """Run the complete preprocessing pipeline.

        Steps: grayscale → Gaussian blur → histogram equalization →
        CLAHE enhancement → histogram computation → Fourier analysis.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.
        output_dir : str, optional
            Directory for saving intermediate outputs.

        Returns
        -------
        dict
            Contains all processed images and statistics.
        """
        import os

        results = {"original_shape": image.shape}

        # 1. Grayscale
        gray = self.to_grayscale(image)
        results["grayscale"] = gray

        # 2. Noise reduction
        blurred = self.gaussian_blur(gray, kernel_size=5)
        results["blurred"] = blurred

        # 3. Histogram equalization
        equalized = self.histogram_equalization(gray)
        results["equalized"] = equalized

        # 4. CLAHE enhancement
        enhanced = self.enhance_contrast(image, method="clahe")
        results["enhanced"] = enhanced

        # 5. Histogram
        hist_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            hist_path = os.path.join(output_dir, "histogram.png")
        hist_data = self.compute_histogram(gray, output_path=hist_path)
        results["histogram"] = hist_data

        # 6. Fourier analysis
        fourier_path = None
        if output_dir:
            fourier_path = os.path.join(output_dir, "fourier_spectrum.png")
        fourier_data = self.fourier_analysis(gray, output_path=fourier_path)
        results["fourier"] = fourier_data

        # Save processed images
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, "grayscale.png"), gray)
            cv2.imwrite(os.path.join(output_dir, "blurred.png"), blurred)
            cv2.imwrite(os.path.join(output_dir, "equalized.png"), equalized)
            cv2.imwrite(os.path.join(output_dir, "enhanced.png"), enhanced)

        results["metadata"] = self.metadata.copy()
        results["processed_image"] = blurred  # Primary output for next stage
        return results
