"""
Report Generator — Structured Output Reports.

Produces a JSON analysis report, a human-readable text summary,
saves annotated output images, and generates a comparison grid.
"""

import json
import os
import datetime
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Report generation engine.

    Converts pipeline results into persistent outputs: a JSON report,
    a plain-text summary, saved images, and a visual comparison grid.
    """

    def __init__(self):
        self.metadata = {}

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    def generate_json_report(self, results: dict,
                             output_path: str) -> str:
        """Serialise pipeline results to a JSON file.

        Non-serialisable objects (numpy arrays, OpenCV contours) are
        automatically converted or stripped.

        Parameters
        ----------
        results : dict
            Pipeline output dictionary.
        output_path : str
            Destination file path.

        Returns
        -------
        str
            The absolute path of the saved file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        clean = self._make_serialisable(results)
        clean["report_generated_at"] = datetime.datetime.now().isoformat()
        clean["report_version"] = "1.0"

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2, ensure_ascii=False)

        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def generate_text_summary(self, results: dict,
                              output_path: str) -> str:
        """Generate a human-readable text summary of the analysis.

        Parameters
        ----------
        results : dict
            Pipeline output dictionary.
        output_path : str
            Destination file path.

        Returns
        -------
        str
            The absolute path of the saved file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        lines = []
        lines.append("=" * 60)
        lines.append("  VisionForge — Analysis Report")
        lines.append("=" * 60)
        lines.append(f"  Generated: {datetime.datetime.now().isoformat()}")
        lines.append("")

        # Input info
        if "input_path" in results:
            lines.append(f"  Input: {results['input_path']}")
        if "original_shape" in results.get("preprocessing", {}):
            shape = results["preprocessing"]["original_shape"]
            lines.append(f"  Image size: {shape}")
        lines.append("")

        # Preprocessing
        if "preprocessing" in results:
            pre = results["preprocessing"]
            lines.append("-" * 60)
            lines.append("  PREPROCESSING")
            lines.append("-" * 60)
            meta = pre.get("metadata", {})
            for key, val in meta.items():
                if key.endswith("_time"):
                    lines.append(f"    {key}: {val:.4f}s")
            hist = pre.get("histogram", {})
            if "mean" in hist:
                lines.append(f"    Mean intensity: {hist['mean']:.2f}")
                lines.append(f"    Std intensity:  {hist['std']:.2f}")
            fourier = pre.get("fourier", {})
            if "dominant_frequency" in fourier:
                lines.append(
                    f"    Dominant frequency: {fourier['dominant_frequency']:.2f}")
            lines.append("")

        # Features
        if "features" in results:
            feat = results["features"]
            lines.append("-" * 60)
            lines.append("  FEATURE EXTRACTION")
            lines.append("-" * 60)
            meta = feat.get("metadata", {})
            for key, val in meta.items():
                if key.endswith("_time"):
                    lines.append(f"    {key}: {val:.4f}s")
            if "harris" in feat:
                lines.append(
                    f"    Harris corners detected: "
                    f"{feat['harris'].get('count', 'N/A')}")
            if "hog" in feat:
                lines.append(
                    f"    HOG descriptor length: "
                    f"{feat['hog'].get('descriptor_length', 'N/A')}")
            canny_ratio = meta.get("canny_edge_ratio")
            if canny_ratio is not None:
                lines.append(f"    Canny edge ratio: {canny_ratio:.4f}")
            lines.append("")

        # Segmentation
        if "segmentation" in results:
            seg = results["segmentation"]
            lines.append("-" * 60)
            lines.append("  SEGMENTATION")
            lines.append("-" * 60)
            lines.append(f"    Method: {seg.get('method', 'N/A')}")
            meta = seg.get("metadata", {})
            for key, val in meta.items():
                if key.endswith("_time"):
                    lines.append(f"    {key}: {val:.4f}s")
                elif "segments" in key or "regions" in key or key == "kmeans_k":
                    lines.append(f"    {key}: {val}")
            lines.append("")

        # Shape analysis
        if "shapes" in results:
            sh = results["shapes"]
            lines.append("-" * 60)
            lines.append("  SHAPE ANALYSIS")
            lines.append("-" * 60)
            if "hough_lines" in sh:
                lines.append(
                    f"    Lines detected: "
                    f"{sh['hough_lines'].get('line_count', 0)}")
            if "hough_circles" in sh:
                lines.append(
                    f"    Circles detected: "
                    f"{sh['hough_circles'].get('circle_count', 0)}")
            if "contours" in sh:
                lines.append(
                    f"    Contours found: "
                    f"{sh['contours'].get('contour_count', 0)}")
            lines.append("")

        # Motion
        if "motion" in results:
            mot = results["motion"]
            lines.append("-" * 60)
            lines.append("  MOTION ANALYSIS")
            lines.append("-" * 60)
            if "dense_optical_flow" in mot:
                mag = mot["dense_optical_flow"].get("magnitude_stats", {})
                lines.append(
                    f"    Dense flow mean magnitude: "
                    f"{mag.get('mean_magnitude', 'N/A')}")
            if "motion_parameters" in mot:
                summary = mot["motion_parameters"].get("summary", {})
                if summary:
                    lines.append(
                        f"    Mean rotation: "
                        f"{summary.get('mean_rotation_deg', 0):.2f}°")
                    lines.append(
                        f"    Mean scale: "
                        f"{summary.get('mean_scale', 1):.4f}")
            lines.append("")

        # Evaluation
        if "evaluation" in results:
            ev = results["evaluation"]
            lines.append("-" * 60)
            lines.append("  EVALUATION")
            lines.append("-" * 60)
            comparison = ev.get("algorithm_comparison", {})
            timing = comparison.get("timing", {})
            for key, val in sorted(timing.items()):
                lines.append(f"    {key}: {val:.4f}s")
            total = comparison.get("total_time")
            if total is not None:
                lines.append(f"    TOTAL: {total:.4f}s")
            lines.append("")

        lines.append("=" * 60)
        lines.append("  End of Report")
        lines.append("=" * 60)

        text = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Save output images
    # ------------------------------------------------------------------

    def save_output_images(self, images_dict: dict,
                           output_dir: str) -> list:
        """Save a dictionary of named images to disk.

        Parameters
        ----------
        images_dict : dict[str, np.ndarray]
            ``{name: image_array}``
        output_dir : str
            Destination directory.

        Returns
        -------
        list[str]
            Paths of saved images.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved = []
        for name, img in images_dict.items():
            if img is None:
                continue
            if not isinstance(img, np.ndarray):
                continue
            path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(path, img)
            saved.append(path)
        return saved

    # ------------------------------------------------------------------
    # Comparison grid
    # ------------------------------------------------------------------

    def generate_comparison_grid(self, images_dict: dict,
                                 output_path: str,
                                 title: str = "VisionForge — Results") -> str:
        """Create a matplotlib grid showing all output images.

        Parameters
        ----------
        images_dict : dict[str, np.ndarray]
            Named images to display.
        output_path : str
            Destination file path.
        title : str
            Grid title.

        Returns
        -------
        str
            Path of the saved grid image.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Filter out None / non-image entries
        valid = {k: v for k, v in images_dict.items()
                 if isinstance(v, np.ndarray) and v.ndim >= 2}

        n = len(valid)
        if n == 0:
            return ""

        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx, (name, img) in enumerate(valid.items()):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            if len(img.shape) == 3 and img.shape[2] == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(name.replace("_", " ").title(), fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_full_report(self, results: dict,
                             output_dir: str) -> dict:
        """Generate JSON report, text summary, and save all images.

        Parameters
        ----------
        results : dict
            Pipeline output dictionary.
        output_dir : str
            Root directory for all report artefacts.

        Returns
        -------
        dict
            ``{"json_path": str, "text_path": str,
               "image_paths": list, "grid_path": str}``
        """
        os.makedirs(output_dir, exist_ok=True)

        # Collect all images for saving and grid
        images = {}
        if "preprocessing" in results:
            pre = results["preprocessing"]
            for key in ("grayscale", "blurred", "equalized", "enhanced"):
                if key in pre and isinstance(pre[key], np.ndarray):
                    images[key] = pre[key]

        if "features" in results:
            feat = results["features"]
            if "canny" in feat and isinstance(feat["canny"], np.ndarray):
                images["canny_edges"] = feat["canny"]
            if "log" in feat and isinstance(feat["log"], np.ndarray):
                images["log_edges"] = feat["log"]
            if "harris" in feat and isinstance(feat["harris"], dict):
                images["harris_corners"] = feat["harris"].get("corner_image")
            if "hog" in feat and isinstance(feat["hog"], dict):
                images["hog_features"] = feat["hog"].get("visualization")

        if "segmentation" in results:
            seg = results["segmentation"]
            if "segmented" in seg:
                images["segmentation"] = seg["segmented"]
            elif "mask" in seg:
                images["segmentation"] = seg["mask"]

        if "shapes" in results:
            sh = results["shapes"]
            if "hough_lines" in sh:
                images["hough_lines"] = sh["hough_lines"].get("annotated_image")
            if "contours" in sh:
                images["contours"] = sh["contours"].get("contour_image")

        # Save images
        img_dir = os.path.join(output_dir, "images")
        image_paths = self.save_output_images(images, img_dir)

        # Comparison grid
        grid_path = self.generate_comparison_grid(
            images, os.path.join(output_dir, "comparison_grid.png"),
        )

        # JSON report
        json_path = self.generate_json_report(
            results, os.path.join(output_dir, "analysis_report.json"),
        )

        # Text summary
        text_path = self.generate_text_summary(
            results, os.path.join(output_dir, "analysis_summary.txt"),
        )

        return {
            "json_path": json_path,
            "text_path": text_path,
            "image_paths": image_paths,
            "grid_path": grid_path,
        }

    # ------------------------------------------------------------------
    # Serialisation helper
    # ------------------------------------------------------------------

    def _make_serialisable(self, obj):
        """Recursively convert non-JSON-serialisable types."""
        if isinstance(obj, dict):
            return {k: self._make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serialisable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            if obj.size > 1000:
                return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, bytes):
            return "<bytes>"
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
