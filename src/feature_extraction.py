"""
Feature Extraction Engine — Module 2: Feature Extraction.

Implements Canny edge detection, Laplacian of Gaussian (LoG),
Difference of Gaussians (DoG), Harris corner detection, HOG descriptor
extraction, scale-space analysis, and Gabor filter bank.
"""

import time
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    Feature extraction engine.

    Provides classical feature-extraction algorithms mapped to Module 2
    of the computer vision syllabus: edge detectors (Canny, LoG, DoG),
    corner detectors (Harris), descriptors (HOG), scale-space analysis,
    and Gabor texture filters.
    """

    def __init__(self):
        self.metadata = {}

    # ------------------------------------------------------------------
    # Edge detectors
    # ------------------------------------------------------------------

    def canny_edges(self, image: np.ndarray,
                    low: int = 50, high: int = 150,
                    auto_threshold: bool = True) -> np.ndarray:
        """Canny edge detector.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        low, high : int
            Lower and upper hysteresis thresholds.
        auto_threshold : bool
            If ``True``, compute thresholds from the median intensity
            (Otsu-style heuristic).

        Returns
        -------
        np.ndarray
            Binary edge map (uint8, 0 or 255).
        """
        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if auto_threshold:
            median = np.median(image)
            low = int(max(0, 0.66 * median))
            high = int(min(255, 1.33 * median))

        edges = cv2.Canny(image, low, high)

        self.metadata["canny_time"] = time.time() - start
        self.metadata["canny_thresholds"] = {"low": low, "high": high}
        self.metadata["canny_edge_pixels"] = int(np.count_nonzero(edges))
        self.metadata["canny_edge_ratio"] = float(
            np.count_nonzero(edges) / edges.size
        )
        return edges

    def log_edges(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Laplacian of Gaussian (LoG) edge / blob detector.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        sigma : float
            Standard deviation of the Gaussian kernel.

        Returns
        -------
        np.ndarray
            Normalised LoG response (uint8).
        """
        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_float = image.astype(np.float64)
        log_response = gaussian_laplace(img_float, sigma=sigma)
        # Scale-normalise
        log_response = log_response * (sigma ** 2)

        # Normalise to uint8 for display
        log_norm = cv2.normalize(np.abs(log_response), None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)

        self.metadata["log_time"] = time.time() - start
        self.metadata["log_sigma"] = sigma
        return log_norm

    def dog_edges(self, image: np.ndarray,
                  sigma1: float = 1.0,
                  sigma2: float = 2.0) -> np.ndarray:
        """Difference of Gaussians (DoG) approximation to LoG.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        sigma1, sigma2 : float
            Standard deviations of the two Gaussian kernels.

        Returns
        -------
        np.ndarray
            Normalised DoG response (uint8).
        """
        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_float = image.astype(np.float64)
        k1 = int(np.ceil(sigma1 * 6)) | 1  # Ensure odd
        k2 = int(np.ceil(sigma2 * 6)) | 1
        blur1 = cv2.GaussianBlur(img_float, (k1, k1), sigma1)
        blur2 = cv2.GaussianBlur(img_float, (k2, k2), sigma2)
        dog = blur1 - blur2

        dog_norm = cv2.normalize(np.abs(dog), None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)

        self.metadata["dog_time"] = time.time() - start
        self.metadata["dog_sigmas"] = {"sigma1": sigma1, "sigma2": sigma2}
        return dog_norm

    # ------------------------------------------------------------------
    # Corner detectors
    # ------------------------------------------------------------------

    def harris_corners(self, image: np.ndarray,
                       block_size: int = 2,
                       ksize: int = 3,
                       k: float = 0.04,
                       threshold_ratio: float = 0.01) -> dict:
        """Harris corner detector.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        block_size : int
            Neighbourhood size for the structure tensor.
        ksize : int
            Sobel aperture.
        k : float
            Harris detector free parameter.
        threshold_ratio : float
            Fraction of max response used as the detection threshold.

        Returns
        -------
        dict
            ``{"corner_image": np.ndarray, "corner_points": list[tuple],
               "count": int, "response_map": np.ndarray}``
        """
        start = time.time()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vis = image.copy()
        else:
            gray = image
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray_f = np.float32(gray)
        response = cv2.cornerHarris(gray_f, block_size, ksize, k)

        # Non-maximum suppression via dilation
        response_dilated = cv2.dilate(response, None)
        threshold = threshold_ratio * response.max()
        corner_mask = (response == response_dilated) & (response > threshold)

        corner_points = list(zip(*np.where(corner_mask)))  # (row, col)
        vis[corner_mask] = [0, 0, 255]  # Mark corners in red

        result = {
            "corner_image": vis,
            "corner_points": corner_points,
            "count": len(corner_points),
            "response_map": response,
        }

        self.metadata["harris_time"] = time.time() - start
        self.metadata["harris_corners_count"] = len(corner_points)
        return result

    # ------------------------------------------------------------------
    # HOG descriptor
    # ------------------------------------------------------------------

    def hog_features(self, image: np.ndarray,
                     orientations: int = 9,
                     pixels_per_cell: tuple = (8, 8),
                     cells_per_block: tuple = (2, 2)) -> dict:
        """Compute Histogram of Oriented Gradients (HOG).

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        orientations : int
            Number of orientation bins.
        pixels_per_cell : tuple
            Cell size in pixels.
        cells_per_block : tuple
            Number of cells per block for normalisation.

        Returns
        -------
        dict
            ``{"descriptor": np.ndarray, "visualization": np.ndarray,
               "descriptor_length": int}``
        """
        from skimage.feature import hog as skimage_hog

        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        descriptor, hog_image = skimage_hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            feature_vector=True,
        )

        # Rescale visualisation to uint8
        hog_vis = (hog_image / hog_image.max() * 255).astype(np.uint8) \
            if hog_image.max() > 0 else hog_image.astype(np.uint8)

        result = {
            "descriptor": descriptor,
            "visualization": hog_vis,
            "descriptor_length": len(descriptor),
        }

        self.metadata["hog_time"] = time.time() - start
        self.metadata["hog_descriptor_length"] = len(descriptor)
        self.metadata["hog_energy"] = float(np.sum(descriptor ** 2))
        return result

    # ------------------------------------------------------------------
    # Scale-space analysis
    # ------------------------------------------------------------------

    def scale_space_analysis(self, image: np.ndarray,
                             num_scales: int = 5,
                             sigma_start: float = 1.0,
                             sigma_factor: float = 1.6) -> dict:
        """Build a Gaussian scale-space and compute LoG at each level.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        num_scales : int
            Number of scales to compute.
        sigma_start : float
            Initial sigma.
        sigma_factor : float
            Multiplicative factor between successive scales.

        Returns
        -------
        dict
            ``{"scales": list[dict], "visualization": np.ndarray}``
            Each scale dict: ``{"sigma": float, "blurred": np.ndarray,
                                "log_response": np.ndarray}``
        """
        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_float = image.astype(np.float64)
        scales = []
        sigma = sigma_start

        for i in range(num_scales):
            k = int(np.ceil(sigma * 6)) | 1
            blurred = cv2.GaussianBlur(img_float, (k, k), sigma)
            log_resp = gaussian_laplace(blurred, sigma=sigma) * (sigma ** 2)
            log_norm = cv2.normalize(np.abs(log_resp), None, 0, 255,
                                     cv2.NORM_MINMAX).astype(np.uint8)
            scales.append({
                "sigma": sigma,
                "blurred": cv2.normalize(blurred, None, 0, 255,
                                         cv2.NORM_MINMAX).astype(np.uint8),
                "log_response": log_norm,
            })
            sigma *= sigma_factor

        # Create a visualisation grid
        n = len(scales)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))
        for i, s in enumerate(scales):
            axes[0, i].imshow(s["blurred"], cmap="gray")
            axes[0, i].set_title(f"σ = {s['sigma']:.2f}")
            axes[0, i].axis("off")
            axes[1, i].imshow(s["log_response"], cmap="hot")
            axes[1, i].set_title(f"LoG σ = {s['sigma']:.2f}")
            axes[1, i].axis("off")
        axes[0, 0].set_ylabel("Gaussian")
        axes[1, 0].set_ylabel("LoG")
        plt.tight_layout()

        # Convert figure to image array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3].copy()  # RGBA → RGB
        plt.close(fig)

        result = {
            "scales": scales,
            "visualization": vis_array,
        }

        self.metadata["scale_space_time"] = time.time() - start
        self.metadata["num_scales"] = num_scales
        return result

    # ------------------------------------------------------------------
    # Gabor filter bank
    # ------------------------------------------------------------------

    def gabor_filter_bank(self, image: np.ndarray,
                          frequencies: list = None,
                          orientations: list = None) -> dict:
        """Apply a bank of Gabor filters to capture texture information.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        frequencies : list[float]
            Spatial frequencies.
        orientations : list[float]
            Orientations in **radians**.

        Returns
        -------
        dict
            ``{"responses": list[dict], "visualization": np.ndarray}``
        """
        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if frequencies is None:
            frequencies = [0.05, 0.1, 0.2, 0.3]
        if orientations is None:
            orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        responses = []
        for freq in frequencies:
            for theta in orientations:
                kernel_size = int(np.ceil(1.0 / freq * 3)) | 1
                if kernel_size < 3:
                    kernel_size = 3
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size),
                    sigma=1.0 / freq * 0.5,
                    theta=theta,
                    lambd=1.0 / freq,
                    gamma=0.5,
                    psi=0,
                )
                filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
                filtered_norm = cv2.normalize(np.abs(filtered), None, 0, 255,
                                              cv2.NORM_MINMAX).astype(np.uint8)
                responses.append({
                    "frequency": freq,
                    "orientation": float(theta),
                    "response": filtered_norm,
                })

        # Visualisation grid
        n_freq = len(frequencies)
        n_orient = len(orientations)
        fig, axes = plt.subplots(n_freq, n_orient,
                                 figsize=(3 * n_orient, 3 * n_freq))
        if n_freq == 1:
            axes = axes[np.newaxis, :]
        if n_orient == 1:
            axes = axes[:, np.newaxis]
        idx = 0
        for fi in range(n_freq):
            for oi in range(n_orient):
                axes[fi, oi].imshow(responses[idx]["response"], cmap="gray")
                axes[fi, oi].set_title(
                    f"f={frequencies[fi]:.2f} θ={np.degrees(orientations[oi]):.0f}°",
                    fontsize=8,
                )
                axes[fi, oi].axis("off")
                idx += 1
        plt.suptitle("Gabor Filter Bank")
        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        vis_array = np.asarray(buf)[:, :, :3].copy()  # RGBA → RGB
        plt.close(fig)

        result = {
            "responses": responses,
            "visualization": vis_array,
        }
        self.metadata["gabor_time"] = time.time() - start
        self.metadata["gabor_num_filters"] = len(responses)
        return result

    # ------------------------------------------------------------------
    # Full feature extraction pipeline
    # ------------------------------------------------------------------

    def run_full_extraction(self, image: np.ndarray,
                            output_dir: str = None) -> dict:
        """Run all feature extractors and return combined results.

        Parameters
        ----------
        image : np.ndarray
            Grayscale or BGR input (grayscale preferred).
        output_dir : str, optional
            Directory for saving output images.

        Returns
        -------
        dict
            Contains all feature maps, descriptors, and statistics.
        """
        results = {}

        # Edges
        results["canny"] = self.canny_edges(image)
        results["log"] = self.log_edges(image, sigma=1.4)
        results["dog"] = self.dog_edges(image, sigma1=1.0, sigma2=2.0)

        # Corners
        results["harris"] = self.harris_corners(image)

        # HOG
        results["hog"] = self.hog_features(image)

        # Scale-space
        results["scale_space"] = self.scale_space_analysis(image)

        # Gabor
        results["gabor"] = self.gabor_filter_bank(image)

        # Save outputs
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "canny_edges.png"),
                        results["canny"])
            cv2.imwrite(os.path.join(output_dir, "log_edges.png"),
                        results["log"])
            cv2.imwrite(os.path.join(output_dir, "dog_edges.png"),
                        results["dog"])
            cv2.imwrite(os.path.join(output_dir, "harris_corners.png"),
                        results["harris"]["corner_image"])
            cv2.imwrite(os.path.join(output_dir, "hog_features.png"),
                        results["hog"]["visualization"])
            cv2.imwrite(os.path.join(output_dir, "scale_space.png"),
                        cv2.cvtColor(results["scale_space"]["visualization"],
                                     cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, "gabor_filters.png"),
                        cv2.cvtColor(results["gabor"]["visualization"],
                                     cv2.COLOR_RGB2BGR))

        results["metadata"] = self.metadata.copy()
        return results
