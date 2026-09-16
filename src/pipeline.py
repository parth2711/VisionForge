"""
Pipeline Controller — Orchestrates the Full CV Analysis Pipeline.

Chains preprocessing → feature extraction → segmentation → shape
analysis → (motion analysis) → evaluation → report generation.
"""

import os
import time
import cv2
import numpy as np

from src.preprocessing import ImagePreprocessor
from src.feature_extraction import FeatureExtractor
from src.segmentation import Segmenter
from src.pattern_analysis import PatternAnalyzer
from src.motion_analysis import MotionAnalyzer
from src.evaluation import Evaluator
from src.report_generator import ReportGenerator


class Pipeline:
    """
    Central pipeline controller for VisionForge.

    Instantiates all engine modules and orchestrates them in the correct
    order, passing intermediate outputs forward.  Supports both single-
    image and video analysis workflows.
    """

    def __init__(self, input_path: str,
                 output_dir: str = "outputs",
                 report_dir: str = "reports",
                 config: dict = None):
        """
        Parameters
        ----------
        input_path : str
            Path to the input image or video file.
        output_dir : str
            Directory for intermediate output images.
        report_dir : str
            Directory for the final report artefacts.
        config : dict, optional
            Pipeline configuration overrides (e.g. segmentation method).
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.config = config or {}

        # Instantiate engines
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.segmenter = Segmenter()
        self.pattern_analyzer = PatternAnalyzer()
        self.motion_analyzer = MotionAnalyzer()
        self.evaluator = Evaluator()
        self.report_generator = ReportGenerator()

    # ------------------------------------------------------------------
    # Input loading
    # ------------------------------------------------------------------

    def _load_image(self) -> np.ndarray:
        """Load an image from *self.input_path*.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        ValueError
            If the file cannot be decoded as an image.
        """
        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(f"File not found: {self.input_path}")
        image = cv2.imread(self.input_path)
        if image is None:
            raise ValueError(
                f"Cannot decode image: {self.input_path}")
        return image

    @staticmethod
    def _is_video(path: str) -> bool:
        """Heuristic check for video file extensions."""
        ext = os.path.splitext(path)[1].lower()
        return ext in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    # ------------------------------------------------------------------
    # Full image analysis
    # ------------------------------------------------------------------

    def run_full_image_analysis(self) -> dict:
        """Execute the complete image-analysis pipeline.

        Stages
        ------
        1. Preprocessing (grayscale, blur, histogram, Fourier)
        2. Feature extraction (Canny, LoG, DoG, Harris, HOG)
        3. Segmentation (configurable method)
        4. Shape analysis (Hough, contours)
        5. Evaluation + report

        Returns
        -------
        dict
            Full pipeline results.
        """
        pipeline_start = time.time()
        results = {"input_path": self.input_path}

        # Load
        image = self._load_image()
        results["original"] = image
        results["original_shape"] = list(image.shape)
        print(f"[VisionForge] Loaded image: {self.input_path} "
              f"({image.shape[1]}x{image.shape[0]})")

        # --- Stage 1: Preprocessing ---
        print("[VisionForge] Stage 1/4 - Preprocessing ...")
        pre_results, pre_time = self.evaluator.measure_processing_time(
            "preprocessing",
            self.preprocessor.run_full_preprocessing,
            image, self.output_dir,
        )
        results["preprocessing"] = pre_results
        print(f"  [OK] Preprocessing complete ({pre_time:.3f}s)")

        # Primary processed image for downstream stages
        processed = pre_results.get("processed_image", image)
        gray = pre_results.get("grayscale", processed)

        # --- Stage 2: Feature Extraction ---
        print("[VisionForge] Stage 2/4 - Feature extraction ...")
        feat_results, feat_time = self.evaluator.measure_processing_time(
            "feature_extraction",
            self.feature_extractor.run_full_extraction,
            gray, self.output_dir,
        )
        results["features"] = feat_results
        print(f"  [OK] Features extracted ({feat_time:.3f}s)")

        # --- Stage 3: Segmentation ---
        seg_method = self.config.get("segmentation_method", "kmeans")
        seg_kwargs = {}
        if seg_method == "kmeans":
            seg_kwargs["k"] = self.config.get("kmeans_k", 5)
        print(f"[VisionForge] Stage 3/4 - Segmentation ({seg_method}) ...")
        seg_results, seg_time = self.evaluator.measure_processing_time(
            "segmentation",
            self.segmenter.run_full_segmentation,
            image, seg_method, self.output_dir, **seg_kwargs,
        )
        results["segmentation"] = seg_results
        print(f"  [OK] Segmentation complete ({seg_time:.3f}s)")

        # --- Stage 4: Shape Analysis ---
        print("[VisionForge] Stage 4/4 - Shape analysis ...")
        edge_map = feat_results.get("canny")
        shape_results, shape_time = self.evaluator.measure_processing_time(
            "shape_analysis",
            self.pattern_analyzer.run_shape_analysis,
            image, edge_map, self.output_dir,
        )
        results["shapes"] = shape_results
        print(f"  [OK] Shape analysis complete ({shape_time:.3f}s)")

        # --- Evaluation ---
        evaluation = self.evaluator.compile_evaluation(results)
        evaluation["total_pipeline_time"] = time.time() - pipeline_start
        results["evaluation"] = evaluation

        # --- Report ---
        print("[VisionForge] Generating report ...")
        report_info = self.report_generator.generate_full_report(
            results, self.report_dir)
        results["report"] = report_info
        print(f"  [OK] Report saved to {self.report_dir}/")

        total = time.time() - pipeline_start
        print(f"\n[VisionForge] Pipeline complete in {total:.3f}s")
        return results

    # ------------------------------------------------------------------
    # Full video analysis
    # ------------------------------------------------------------------

    def run_full_video_analysis(self) -> dict:
        """Execute image pipeline on the first frame + full motion analysis.

        Returns
        -------
        dict
            Combined image + motion results.
        """
        pipeline_start = time.time()

        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(f"File not found: {self.input_path}")

        # Extract first frame for static analysis
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Cannot read first frame from video.")

        results = {"input_path": self.input_path}
        results["original"] = first_frame
        results["original_shape"] = list(first_frame.shape)
        print(f"[VisionForge] Loaded video: {self.input_path} "
              f"(first frame {first_frame.shape[1]}x{first_frame.shape[0]})")

        # Static analysis on first frame (stages 1-4)
        print("[VisionForge] Running static analysis on first frame ...")

        # Stage 1
        pre_results = self.preprocessor.run_full_preprocessing(
            first_frame, self.output_dir)
        results["preprocessing"] = pre_results

        processed = pre_results.get("processed_image", first_frame)
        gray = pre_results.get("grayscale", processed)

        # Stage 2
        feat_results = self.feature_extractor.run_full_extraction(
            gray, self.output_dir)
        results["features"] = feat_results

        # Stage 3
        seg_method = self.config.get("segmentation_method", "kmeans")
        seg_results = self.segmenter.run_full_segmentation(
            first_frame, seg_method, self.output_dir)
        results["segmentation"] = seg_results

        # Stage 4
        edge_map = feat_results.get("canny")
        shape_results = self.pattern_analyzer.run_shape_analysis(
            first_frame, edge_map, self.output_dir)
        results["shapes"] = shape_results

        print("  [OK] Static analysis complete")

        # --- Stage 5: Motion Analysis ---
        print("[VisionForge] Running motion analysis ...")
        motion_results, motion_time = self.evaluator.measure_processing_time(
            "motion_analysis",
            self.motion_analyzer.run_full_motion_analysis,
            self.input_path, self.output_dir,
        )
        results["motion"] = motion_results
        print(f"  [OK] Motion analysis complete ({motion_time:.3f}s)")

        # Evaluation + Report
        evaluation = self.evaluator.compile_evaluation(results)
        evaluation["total_pipeline_time"] = time.time() - pipeline_start
        results["evaluation"] = evaluation

        print("[VisionForge] Generating report ...")
        report_info = self.report_generator.generate_full_report(
            results, self.report_dir)
        results["report"] = report_info
        print(f"  [OK] Report saved to {self.report_dir}/")

        total = time.time() - pipeline_start
        print(f"\n[VisionForge] Pipeline complete in {total:.3f}s")
        return results

    # ------------------------------------------------------------------
    # Single-stage runners (used by CLI subcommands)
    # ------------------------------------------------------------------

    def run_stage(self, stage_name: str, **kwargs) -> dict:
        """Run a single pipeline stage.

        Parameters
        ----------
        stage_name : str
            One of ``'preprocess'``, ``'edges'``, ``'features'``,
            ``'segment'``, ``'shapes'``, ``'motion'``.

        Returns
        -------
        dict
            Stage results.
        """
        dispatch = {
            "preprocess": self._run_preprocess,
            "edges": self._run_edges,
            "features": self._run_features,
            "segment": self._run_segment,
            "shapes": self._run_shapes,
            "motion": self._run_motion,
        }
        if stage_name not in dispatch:
            raise ValueError(
                f"Unknown stage '{stage_name}'. "
                f"Choose from: {', '.join(dispatch.keys())}")
        return dispatch[stage_name](**kwargs)

    def _run_preprocess(self, **kwargs) -> dict:
        image = self._load_image()
        return self.preprocessor.run_full_preprocessing(
            image, self.output_dir)

    def _run_edges(self, method: str = "canny", **kwargs) -> dict:
        image = self._load_image()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image
        gray = self.preprocessor.gaussian_blur(gray, 5)

        methods = {
            "canny": lambda: self.feature_extractor.canny_edges(gray),
            "log": lambda: self.feature_extractor.log_edges(gray),
            "dog": lambda: self.feature_extractor.dog_edges(gray),
        }
        if method not in methods:
            raise ValueError(
                f"Unknown edge method '{method}'. "
                f"Choose from: {', '.join(methods.keys())}")

        edges = methods[method]()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{method}_edges.png")
        cv2.imwrite(path, edges)
        print(f"[VisionForge] {method} edges saved to {path}")
        return {"edges": edges, "method": method, "output_path": path}

    def _run_features(self, method: str = "harris", **kwargs) -> dict:
        image = self._load_image()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image

        methods = {
            "harris": lambda: self.feature_extractor.harris_corners(gray),
            "hog": lambda: self.feature_extractor.hog_features(gray),
            "scale-space": lambda: self.feature_extractor.scale_space_analysis(
                gray),
            "gabor": lambda: self.feature_extractor.gabor_filter_bank(gray),
        }
        if method not in methods:
            raise ValueError(
                f"Unknown feature method '{method}'. "
                f"Choose from: {', '.join(methods.keys())}")

        result = methods[method]()
        os.makedirs(self.output_dir, exist_ok=True)

        # Save output
        if method == "harris" and "corner_image" in result:
            path = os.path.join(self.output_dir, "harris_corners.png")
            cv2.imwrite(path, result["corner_image"])
            print(f"[VisionForge] Harris corners saved to {path}")
        elif method == "hog" and "visualization" in result:
            path = os.path.join(self.output_dir, "hog_features.png")
            cv2.imwrite(path, result["visualization"])
            print(f"[VisionForge] HOG features saved to {path}")
        elif "visualization" in result:
            path = os.path.join(self.output_dir, f"{method}_features.png")
            vis = result["visualization"]
            if len(vis.shape) == 3 and vis.shape[2] == 3:
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, vis)
            print(f"[VisionForge] {method} features saved to {path}")

        return {"result": result, "method": method}

    def _run_segment(self, method: str = "kmeans", **kwargs) -> dict:
        image = self._load_image()
        return self.segmenter.run_full_segmentation(
            image, method, self.output_dir, **kwargs)

    def _run_shapes(self, **kwargs) -> dict:
        image = self._load_image()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image
        edges = self.feature_extractor.canny_edges(gray)
        return self.pattern_analyzer.run_shape_analysis(
            image, edges, self.output_dir)

    def _run_motion(self, method: str = "all", **kwargs) -> dict:
        if not self._is_video(self.input_path):
            raise ValueError(
                "Motion analysis requires a video file. "
                f"Got: {self.input_path}")

        methods = {
            "background": lambda: self.motion_analyzer.background_subtraction(
                self.input_path),
            "optical-flow": lambda: self.motion_analyzer.optical_flow_dense(
                self.input_path),
            "sparse-flow": lambda: self.motion_analyzer.optical_flow_sparse(
                self.input_path),
            "motion-params": lambda: self.motion_analyzer.estimate_motion_parameters(
                self.input_path),
            "all": lambda: self.motion_analyzer.run_full_motion_analysis(
                self.input_path, self.output_dir),
        }
        if method not in methods:
            raise ValueError(
                f"Unknown motion method '{method}'. "
                f"Choose from: {', '.join(methods.keys())}")
        return methods[method]()
