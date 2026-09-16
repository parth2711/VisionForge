#!/usr/bin/env python3
"""
VisionForge — Computer Vision Image & Scene Analysis Toolkit.

Command-line entry point.  Run ``python main.py --help`` for usage.
"""

import argparse
import sys
import os

from src.pipeline import Pipeline


# ──────────────────────────────────────────────────────────────────────
# CLI definition
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="visionforge",
        description=(
            "VisionForge — A modular classical computer vision pipeline.\n"
            "Run preprocessing, feature extraction, segmentation, shape\n"
            "analysis, and motion analysis from the command line."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage")

    # ── preprocess ────────────────────────────────────────────────────
    sp = subparsers.add_parser(
        "preprocess",
        help="Run the preprocessing pipeline (grayscale, blur, histogram, "
             "Fourier analysis).",
    )
    _add_common_args(sp)

    # ── edges ─────────────────────────────────────────────────────────
    sp = subparsers.add_parser(
        "edges",
        help="Run edge detection on an image.",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--method", choices=["canny", "log", "dog"],
        default="canny", help="Edge detection algorithm (default: canny).",
    )

    # ── features ──────────────────────────────────────────────────────
    sp = subparsers.add_parser(
        "features",
        help="Extract features from an image.",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--method", choices=["harris", "hog", "scale-space", "gabor"],
        default="harris",
        help="Feature extraction algorithm (default: harris).",
    )

    # ── segment ───────────────────────────────────────────────────────
    sp = subparsers.add_parser(
        "segment",
        help="Segment an image into regions.",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--method",
        choices=["kmeans", "meanshift", "region", "edge", "watershed"],
        default="kmeans",
        help="Segmentation algorithm (default: kmeans).",
    )
    sp.add_argument(
        "--k", type=int, default=5,
        help="Number of clusters for K-Means (default: 5).",
    )

    # ── shapes ────────────────────────────────────────────────────────
    sp = subparsers.add_parser(
        "shapes",
        help="Detect lines, circles, and contours.",
    )
    _add_common_args(sp)

    # ── motion ────────────────────────────────────────────────────────
    sp = subparsers.add_parser(
        "motion",
        help="Run motion analysis on a video.",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--method",
        choices=["background", "optical-flow", "sparse-flow",
                 "motion-params", "all"],
        default="all",
        help="Motion analysis method (default: all).",
    )

    # ── analyze (full image pipeline) ─────────────────────────────────
    sp = subparsers.add_parser(
        "analyze",
        help="Run the complete analysis pipeline on an image.",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--seg-method",
        choices=["kmeans", "meanshift", "region", "edge", "watershed"],
        default="kmeans",
        help="Segmentation method for the full pipeline (default: kmeans).",
    )
    sp.add_argument(
        "--k", type=int, default=5,
        help="Number of clusters for K-Means segmentation (default: 5).",
    )

    # ── analyze-video (full video pipeline) ───────────────────────────
    sp = subparsers.add_parser(
        "analyze-video",
        help="Run the complete analysis pipeline on a video.",
    )
    _add_common_args(sp)
    sp.add_argument(
        "--seg-method",
        choices=["kmeans", "meanshift", "region", "edge", "watershed"],
        default="kmeans",
        help="Segmentation method for the full pipeline (default: kmeans).",
    )

    return parser


def _add_common_args(subparser: argparse.ArgumentParser) -> None:
    """Add arguments common to every subcommand."""
    subparser.add_argument(
        "--input", required=True,
        help="Path to the input image or video file.",
    )
    subparser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for output images (default: outputs/).",
    )
    subparser.add_argument(
        "--report-dir", default="reports",
        help="Directory for report artefacts (default: reports/).",
    )
    subparser.add_argument(
        "--report", action="store_true",
        help="Generate a JSON analysis report.",
    )


# ──────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ──────────────────────────────────────────────────────────────────────

def cmd_preprocess(args) -> None:
    pipeline = _make_pipeline(args)
    results = pipeline.run_stage("preprocess")
    print(f"[VisionForge] Preprocessing complete.  Outputs in: {args.output_dir}/")
    if args.report:
        pipeline.report_generator.generate_full_report(
            {"preprocessing": results}, args.report_dir)
        print(f"[VisionForge] Report saved to: {args.report_dir}/")


def cmd_edges(args) -> None:
    pipeline = _make_pipeline(args)
    pipeline.run_stage("edges", method=args.method)


def cmd_features(args) -> None:
    pipeline = _make_pipeline(args)
    pipeline.run_stage("features", method=args.method)


def cmd_segment(args) -> None:
    pipeline = _make_pipeline(args)
    kwargs = {}
    if args.method == "kmeans":
        kwargs["k"] = args.k
    results = pipeline.run_stage("segment", method=args.method, **kwargs)
    method_label = args.method
    print(f"[VisionForge] Segmentation ({method_label}) complete.  "
          f"Outputs in: {args.output_dir}/")
    if args.report:
        pipeline.report_generator.generate_full_report(
            {"segmentation": results}, args.report_dir)
        print(f"[VisionForge] Report saved to: {args.report_dir}/")


def cmd_shapes(args) -> None:
    pipeline = _make_pipeline(args)
    results = pipeline.run_stage("shapes")
    hl = results.get("hough_lines", {}).get("line_count", 0)
    hc = results.get("hough_circles", {}).get("circle_count", 0)
    cc = results.get("contours", {}).get("contour_count", 0)
    print(f"[VisionForge] Shape analysis complete: "
          f"{hl} lines, {hc} circles, {cc} contours.")


def cmd_motion(args) -> None:
    pipeline = _make_pipeline(args)
    pipeline.run_stage("motion", method=args.method)
    print(f"[VisionForge] Motion analysis complete.  "
          f"Outputs in: {args.output_dir}/")


def cmd_analyze(args) -> None:
    config = {
        "segmentation_method": args.seg_method,
        "kmeans_k": args.k,
    }
    pipeline = Pipeline(
        args.input, args.output_dir, args.report_dir, config,
    )
    pipeline.run_full_image_analysis()


def cmd_analyze_video(args) -> None:
    config = {
        "segmentation_method": args.seg_method,
    }
    pipeline = Pipeline(
        args.input, args.output_dir, args.report_dir, config,
    )
    pipeline.run_full_video_analysis()


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_pipeline(args) -> Pipeline:
    """Create a Pipeline instance from parsed CLI arguments."""
    return Pipeline(args.input, args.output_dir, args.report_dir)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

COMMAND_MAP = {
    "preprocess": cmd_preprocess,
    "edges": cmd_edges,
    "features": cmd_features,
    "segment": cmd_segment,
    "shapes": cmd_shapes,
    "motion": cmd_motion,
    "analyze": cmd_analyze,
    "analyze-video": cmd_analyze_video,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
