"""
Segmentation Engine — Module 3: Image Segmentation.

Implements K-Means clustering, Mean-Shift filtering, region growing,
edge-based segmentation, and watershed segmentation.
"""

import time
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Segmenter:
    """
    Image segmentation engine.

    Provides multiple segmentation strategies: colour-space clustering
    (K-Means, Mean-Shift), region growing, edge-based segmentation,
    and marker-based watershed.
    """

    def __init__(self):
        self.metadata = {}

    # ------------------------------------------------------------------
    # K-Means segmentation
    # ------------------------------------------------------------------

    def kmeans_segmentation(self, image: np.ndarray,
                            k: int = 5,
                            max_iter: int = 100) -> dict:
        """Segment an image into *k* colour clusters using K-Means.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.
        k : int
            Number of clusters.
        max_iter : int
            Maximum iterations for K-Means.

        Returns
        -------
        dict
            ``{"segmented": np.ndarray, "labels": np.ndarray,
               "centers": np.ndarray, "k": int, "inertia": float}``
        """
        start = time.time()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w, c = image.shape
        pixels = image.reshape(-1, c).astype(np.float32)

        km = KMeans(n_clusters=k, max_iter=max_iter, n_init=10,
                    random_state=42)
        labels = km.fit_predict(pixels)
        centers = km.cluster_centers_.astype(np.uint8)

        segmented = centers[labels].reshape(h, w, c)

        result = {
            "segmented": segmented,
            "labels": labels.reshape(h, w),
            "centers": centers,
            "k": k,
            "inertia": float(km.inertia_),
        }

        self.metadata["kmeans_time"] = time.time() - start
        self.metadata["kmeans_k"] = k
        self.metadata["kmeans_inertia"] = float(km.inertia_)
        return result

    # ------------------------------------------------------------------
    # Mean-Shift segmentation
    # ------------------------------------------------------------------

    def meanshift_segmentation(self, image: np.ndarray,
                               spatial_radius: int = 20,
                               color_radius: int = 40,
                               max_level: int = 1) -> dict:
        """Segment an image using Mean-Shift filtering.

        Uses ``cv2.pyrMeanShiftFiltering`` which jointly clusters
        pixels by spatial proximity and colour similarity.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.
        spatial_radius : int
            Spatial bandwidth (window size).
        color_radius : int
            Colour bandwidth.
        max_level : int
            Maximum pyramid level for the segmentation.

        Returns
        -------
        dict
            ``{"segmented": np.ndarray, "num_segments": int}``
        """
        start = time.time()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        shifted = cv2.pyrMeanShiftFiltering(image, spatial_radius,
                                            color_radius, maxLevel=max_level)

        # Count unique colours as a proxy for segment count
        unique_colors = np.unique(shifted.reshape(-1, 3), axis=0)

        result = {
            "segmented": shifted,
            "num_segments": len(unique_colors),
        }

        self.metadata["meanshift_time"] = time.time() - start
        self.metadata["meanshift_segments"] = len(unique_colors)
        self.metadata["meanshift_spatial_radius"] = spatial_radius
        self.metadata["meanshift_color_radius"] = color_radius
        return result

    # ------------------------------------------------------------------
    # Region growing
    # ------------------------------------------------------------------

    def region_growing(self, image: np.ndarray,
                       seed_point: tuple = None,
                       threshold: int = 10) -> dict:
        """Segment a region by growing from a seed pixel.

        Uses a simple BFS flood-fill approach: starting from the seed,
        recursively add neighbouring pixels whose intensity is within
        *threshold* of the seed intensity.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input image.
        seed_point : tuple, optional
            (row, col) seed.  Defaults to the image centre.
        threshold : int
            Maximum intensity difference for inclusion.

        Returns
        -------
        dict
            ``{"mask": np.ndarray, "region_size": int,
               "seed_point": tuple, "seed_intensity": int}``
        """
        start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape
        if seed_point is None:
            seed_point = (h // 2, w // 2)

        seed_val = int(image[seed_point[0], seed_point[1]])
        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)

        stack = [seed_point]
        visited[seed_point[0], seed_point[1]] = True

        while stack:
            r, c = stack.pop()
            if abs(int(image[r, c]) - seed_val) <= threshold:
                mask[r, c] = 255
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

        result = {
            "mask": mask,
            "region_size": int(np.count_nonzero(mask)),
            "seed_point": seed_point,
            "seed_intensity": seed_val,
        }

        self.metadata["region_growing_time"] = time.time() - start
        self.metadata["region_growing_size"] = result["region_size"]
        return result

    # ------------------------------------------------------------------
    # Edge-based segmentation
    # ------------------------------------------------------------------

    def edge_based_segmentation(self, image: np.ndarray) -> dict:
        """Segment an image using edge detection + morphological closing.

        Workflow: Canny edges → morphological closing → contour finding →
        fill contours to create labelled regions.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or BGR).

        Returns
        -------
        dict
            ``{"segmented": np.ndarray, "edges": np.ndarray,
               "num_regions": int}``
        """
        start = time.time()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Morphological closing to connect edge fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,
                                  iterations=2)

        # Find contours and fill to create regions
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        label_map = np.zeros(gray.shape, dtype=np.int32)
        for i, cnt in enumerate(contours, start=1):
            cv2.drawContours(label_map, [cnt], -1, i, thickness=cv2.FILLED)

        # Coloured visualisation
        num_regions = len(contours)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_regions, 1)))
        vis = np.zeros((*gray.shape, 3), dtype=np.uint8)
        for i in range(1, num_regions + 1):
            color_idx = (i - 1) % len(colors)
            vis[label_map == i] = (np.array(colors[color_idx][:3]) * 255).astype(np.uint8)

        result = {
            "segmented": vis,
            "edges": edges,
            "label_map": label_map,
            "num_regions": num_regions,
        }

        self.metadata["edge_seg_time"] = time.time() - start
        self.metadata["edge_seg_regions"] = num_regions
        return result

    # ------------------------------------------------------------------
    # Watershed segmentation
    # ------------------------------------------------------------------

    def watershed_segmentation(self, image: np.ndarray) -> dict:
        """Marker-based watershed segmentation.

        Automatically computes markers via distance transform + thresholding.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.

        Returns
        -------
        dict
            ``{"segmented": np.ndarray, "markers": np.ndarray,
               "num_segments": int}``
        """
        start = time.time()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                                   iterations=2)

        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure foreground via distance transform
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed
        markers = cv2.watershed(image, markers)
        vis = image.copy()
        vis[markers == -1] = [0, 0, 255]  # Boundaries in red

        result = {
            "segmented": vis,
            "markers": markers,
            "num_segments": num_labels,
        }

        self.metadata["watershed_time"] = time.time() - start
        self.metadata["watershed_segments"] = num_labels
        return result

    # ------------------------------------------------------------------
    # Full segmentation pipeline
    # ------------------------------------------------------------------

    def run_full_segmentation(self, image: np.ndarray,
                              method: str = "kmeans",
                              output_dir: str = None,
                              **kwargs) -> dict:
        """Run a selected segmentation method.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        method : str
            One of ``'kmeans'``, ``'meanshift'``, ``'region'``,
            ``'edge'``, ``'watershed'``.
        output_dir : str, optional
            Directory for saving output images.
        **kwargs
            Forwarded to the chosen method.

        Returns
        -------
        dict
            Segmentation results.
        """
        dispatch = {
            "kmeans": lambda: self.kmeans_segmentation(image, **kwargs),
            "meanshift": lambda: self.meanshift_segmentation(image, **kwargs),
            "region": lambda: self.region_growing(image, **kwargs),
            "edge": lambda: self.edge_based_segmentation(image),
            "watershed": lambda: self.watershed_segmentation(image),
        }

        if method not in dispatch:
            raise ValueError(
                f"Unknown segmentation method '{method}'. "
                f"Choose from: {', '.join(dispatch.keys())}"
            )

        result = dispatch[method]()
        result["method"] = method

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            seg_key = "segmented" if "segmented" in result else "mask"
            seg_img = result[seg_key]
            cv2.imwrite(os.path.join(output_dir,
                                     f"segmentation_{method}.png"), seg_img)

        result["metadata"] = self.metadata.copy()
        return result
