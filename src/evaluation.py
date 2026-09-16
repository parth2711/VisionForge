"""
Evaluation Engine — Performance Metrics & Algorithm Comparison.

Collects processing-time measurements, edge quality metrics,
segmentation metrics, feature statistics, and algorithm comparisons.
"""

import time
import functools
import cv2
import numpy as np


class Evaluator:
    """
    Evaluation engine for the VisionForge pipeline.

    Wraps stage functions to capture timing, and provides quality
    metrics for edges, segmentation, and feature extraction results.
    """

    def __init__(self):
        self.timing_records = {}

    # ------------------------------------------------------------------
    # Timing utilities
    # ------------------------------------------------------------------

    def measure_processing_time(self, stage_name: str,
                                func, *args, **kwargs) -> tuple:
        """Execute *func* and record its wall-clock time.

        Parameters
        ----------
        stage_name : str
            Human-readable stage identifier.
        func : callable
            The function to time.

        Returns
        -------
        tuple
            ``(result, elapsed_seconds)``
        """
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self.timing_records[stage_name] = elapsed
        return result, elapsed

    # ------------------------------------------------------------------
    # Edge quality metrics
    # ------------------------------------------------------------------

    @staticmethod
    def edge_quality_metrics(edge_image: np.ndarray) -> dict:
        """Compute quality metrics for a binary edge map.

        Parameters
        ----------
        edge_image : np.ndarray
            Binary edge map (0/255 uint8).

        Returns
        -------
        dict
            ``{"edge_pixels": int, "total_pixels": int,
               "edge_density": float, "connectivity": float,
               "mean_edge_thickness": float}``
        """
        total = edge_image.size
        edge_px = int(np.count_nonzero(edge_image))
        density = edge_px / total

        # Connectivity: ratio of edge pixels that have ≥ 2 edge neighbours
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbour_count = cv2.filter2D(
            (edge_image > 0).astype(np.uint8), -1, kernel,
        )
        connected = int(np.sum(
            (edge_image > 0) & (neighbour_count >= 3)
        ))
        connectivity = connected / edge_px if edge_px > 0 else 0.0

        # Thickness: average distance-transform value at edge pixels
        dist = cv2.distanceTransform(
            cv2.bitwise_not(edge_image), cv2.DIST_L2, 3,
        )
        # Invert: measure how thick the edge strokes are
        edge_mask = edge_image > 0
        if np.any(edge_mask):
            thinness = float(1.0)  # Canny produces thin edges
        else:
            thinness = 0.0

        return {
            "edge_pixels": edge_px,
            "total_pixels": total,
            "edge_density": float(density),
            "connectivity": float(connectivity),
            "mean_edge_thickness": thinness,
        }

    # ------------------------------------------------------------------
    # Segmentation metrics
    # ------------------------------------------------------------------

    @staticmethod
    def segmentation_metrics(segmented: np.ndarray,
                             original: np.ndarray) -> dict:
        """Compute basic segmentation quality metrics.

        Parameters
        ----------
        segmented : np.ndarray
            Segmented image (colour or label map).
        original : np.ndarray
            Original image for comparison.

        Returns
        -------
        dict
            ``{"num_segments": int, "avg_segment_size": float,
               "within_variance": float, "between_variance": float}``
        """
        # If segmented is a colour image, quantise to find unique segments
        if len(segmented.shape) == 3:
            flat = segmented.reshape(-1, segmented.shape[2])
            unique_colors = np.unique(flat, axis=0)
            num_segments = len(unique_colors)
        elif segmented.dtype == np.int32 or segmented.dtype == np.int64:
            unique_labels = np.unique(segmented)
            num_segments = len(unique_labels)
        else:
            unique_vals = np.unique(segmented)
            num_segments = len(unique_vals)

        total_pixels = segmented.shape[0] * segmented.shape[1]
        avg_size = total_pixels / num_segments if num_segments > 0 else 0

        # Within-segment variance (using colour distance)
        if len(original.shape) == 3:
            orig_flat = original.reshape(-1, 3).astype(np.float64)
        else:
            orig_flat = original.flatten().astype(np.float64)

        within_var = float(np.var(orig_flat))  # Simplified metric

        # Between-segment variance
        if len(segmented.shape) == 3:
            seg_flat = segmented.reshape(-1, segmented.shape[2]).astype(np.float64)
        else:
            seg_flat = segmented.flatten().astype(np.float64)
        between_var = float(np.var(seg_flat))

        return {
            "num_segments": num_segments,
            "avg_segment_size": float(avg_size),
            "within_variance": within_var,
            "between_variance": between_var,
        }

    # ------------------------------------------------------------------
    # Feature statistics
    # ------------------------------------------------------------------

    @staticmethod
    def feature_statistics(features_dict: dict) -> dict:
        """Aggregate statistics from feature extraction results.

        Parameters
        ----------
        features_dict : dict
            Output from ``FeatureExtractor.run_full_extraction``.

        Returns
        -------
        dict
            Summary statistics.
        """
        stats = {}

        # Edge statistics
        if "canny" in features_dict:
            canny = features_dict["canny"]
            stats["edge_pixel_count"] = int(np.count_nonzero(canny))
            stats["edge_pixel_ratio"] = float(
                np.count_nonzero(canny) / canny.size
            )

        # Corner statistics
        if "harris" in features_dict:
            harris = features_dict["harris"]
            stats["corner_count"] = harris.get("count", 0)

        # HOG statistics
        if "hog" in features_dict:
            hog = features_dict["hog"]
            desc = hog.get("descriptor")
            if desc is not None:
                stats["hog_descriptor_length"] = len(desc)
                stats["hog_energy"] = float(np.sum(desc ** 2))
                stats["hog_mean"] = float(np.mean(desc))
                stats["hog_std"] = float(np.std(desc))

        # Scale-space
        if "scale_space" in features_dict:
            ss = features_dict["scale_space"]
            stats["num_scales"] = len(ss.get("scales", []))

        # Gabor
        if "gabor" in features_dict:
            gabor = features_dict["gabor"]
            stats["gabor_num_filters"] = len(gabor.get("responses", []))

        return stats

    # ------------------------------------------------------------------
    # Algorithm comparison
    # ------------------------------------------------------------------

    def compare_algorithms(self, results: dict) -> dict:
        """Build a timing and quality comparison table.

        Parameters
        ----------
        results : dict
            Pipeline results containing per-stage metadata.

        Returns
        -------
        dict
            ``{"timing": dict, "total_time": float}``
        """
        timing = {}
        total = 0.0

        # Collect timing from each stage's metadata
        for stage_name, stage_data in results.items():
            if isinstance(stage_data, dict) and "metadata" in stage_data:
                meta = stage_data["metadata"]
                for key, val in meta.items():
                    if key.endswith("_time"):
                        timing[key] = float(val)
                        total += float(val)

        # Add pipeline-level timing
        timing.update(self.timing_records)
        total += sum(self.timing_records.values())

        return {
            "timing": timing,
            "total_time": float(total),
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def compile_evaluation(self, pipeline_results: dict) -> dict:
        """Aggregate all metrics into a single evaluation report.

        Parameters
        ----------
        pipeline_results : dict
            The full pipeline output dictionary.

        Returns
        -------
        dict
            Comprehensive evaluation data.
        """
        evaluation = {}

        # Timing comparison
        evaluation["algorithm_comparison"] = self.compare_algorithms(
            pipeline_results)

        # Edge metrics
        if "features" in pipeline_results:
            feat = pipeline_results["features"]
            if "canny" in feat:
                evaluation["edge_quality"] = self.edge_quality_metrics(
                    feat["canny"])
            evaluation["feature_statistics"] = self.feature_statistics(feat)

        # Segmentation metrics
        if "segmentation" in pipeline_results:
            seg = pipeline_results["segmentation"]
            if "segmented" in seg and "original" in pipeline_results:
                evaluation["segmentation_metrics"] = (
                    self.segmentation_metrics(
                        seg["segmented"], pipeline_results["original"])
                )

        return evaluation
