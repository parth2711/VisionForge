"""
VisionForge — Computer Vision Image & Scene Analysis Toolkit.

A modular, CLI-driven classical computer vision pipeline that chains
preprocessing, feature extraction, segmentation, shape analysis, and
motion analysis stages, producing annotated output images and structured
JSON reports.
"""

from src.preprocessing import ImagePreprocessor
from src.transformations import GeometricTransformer
from src.feature_extraction import FeatureExtractor
from src.segmentation import Segmenter
from src.pattern_analysis import PatternAnalyzer
from src.motion_analysis import MotionAnalyzer
from src.evaluation import Evaluator
from src.report_generator import ReportGenerator
from src.pipeline import Pipeline

__all__ = [
    "ImagePreprocessor",
    "GeometricTransformer",
    "FeatureExtractor",
    "Segmenter",
    "PatternAnalyzer",
    "MotionAnalyzer",
    "Evaluator",
    "ReportGenerator",
    "Pipeline",
]

__version__ = "1.0.0"
