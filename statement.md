# VisionForge — Problem Statement

## Title

**VisionForge: Computer Vision Image & Scene Analysis Toolkit**

## Objective

Design and implement a modular, command-line computer vision system that takes an image or video as input and performs multiple stages of classical computer vision analysis through a unified pipeline.

## Problem Description

Digital image processing and computer vision form the backbone of modern visual computing applications — from medical imaging and autonomous vehicles to surveillance and quality inspection. Understanding the fundamental algorithms behind these systems is essential for any computer science practitioner.

This project implements a **complete classical computer vision pipeline** that processes input images and videos through five interconnected stages:

1. **Preprocessing** — Convert to grayscale, reduce noise, equalise histograms, analyse frequency content via Fourier Transform, and enhance contrast.

2. **Feature Extraction** — Detect edges (Canny, LoG, DoG), corners (Harris), compute texture descriptors (HOG, Gabor), and build scale-space representations.

3. **Segmentation** — Partition images into meaningful regions using K-Means clustering, Mean-Shift filtering, region growing, edge-based segmentation, and watershed methods.

4. **Shape & Pattern Analysis** — Detect geometric primitives (Hough Transform for lines and circles), extract contour properties (area, perimeter, solidity, circularity), and perform dimensionality reduction (PCA) and classification (KNN).

5. **Motion Analysis** — For video inputs, perform background subtraction (MOG2/KNN), compute dense and sparse optical flow (Farneback, Lucas-Kanade/KLT), and estimate global motion parameters (translation, rotation, scale).

The output of each stage flows into the next, making this a true pipeline rather than a collection of disconnected algorithms. The system produces annotated output images and a structured JSON analysis report.

## Scope

### In Scope
- Classical image processing (filtering, transforms, histogram operations)
- Classical feature extraction (edges, corners, descriptors)
- Multiple segmentation algorithms with user-selectable methods
- Shape detection and contour analysis
- Video motion analysis (background subtraction, optical flow)
- Performance evaluation and algorithm comparison
- Structured report generation (JSON + text summary)
- Command-line interface with subcommands

### Out of Scope
- Deep learning / neural network-based approaches (e.g. YOLO, CNNs)
- Real-time video processing / GUI
- GPU acceleration
- 3D reconstruction / stereo vision

## Functional Requirements

1. The system shall accept image files (JPEG, PNG) and video files (MP4, AVI) as input.
2. The system shall provide individual CLI commands for each pipeline stage.
3. The system shall provide a single command to run the complete analysis pipeline.
4. The system shall save all output images to a configurable output directory.
5. The system shall generate a JSON analysis report with measurements and statistics.
6. The system shall generate a human-readable text summary of the analysis.
7. The system shall allow the user to select specific algorithms for segmentation and feature extraction.
8. The system shall measure and report processing time for each stage.

## Non-Functional Requirements

1. **Performance** — Processing time is measured per stage; unnecessary image copies are avoided.
2. **Usability** — Simple CLI with descriptive help text and meaningful error messages.
3. **Reliability** — Input validation, exception handling, and graceful error recovery.
4. **Maintainability** — Modular Python architecture with each algorithm in its own method.
5. **Scalability** — Algorithms are separated into independent, swappable modules.
6. **Resource Efficiency** — Processing parameters are configurable (resolution, cluster count, threshold values).

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Image Processing | OpenCV (`opencv-python`) |
| Array Operations | NumPy |
| Machine Learning | scikit-learn (K-Means, KNN, PCA) |
| Image Algorithms | scikit-image (HOG, region processing) |
| Signal Processing | SciPy (Gaussian Laplacian) |
| Visualisation | Matplotlib |
| Testing | pytest |

## Modules

| # | Module | File | Responsibility |
|---|---|---|---|
| 1 | Preprocessing Engine | `src/preprocessing.py` | Grayscale, blur, histogram, Fourier, enhancement |
| 2 | Geometric Transformer | `src/transformations.py` | Euclidean, affine, projective transforms |
| 3 | Feature Extractor | `src/feature_extraction.py` | Canny, LoG, DoG, Harris, HOG, scale-space, Gabor |
| 4 | Segmenter | `src/segmentation.py` | K-Means, Mean-Shift, region growing, edge-based, watershed |
| 5 | Pattern Analyzer | `src/pattern_analysis.py` | Hough lines/circles, contours, PCA, KNN |
| 6 | Motion Analyzer | `src/motion_analysis.py` | Background subtraction, optical flow, motion estimation |
| 7 | Evaluator | `src/evaluation.py` | Timing, quality metrics, algorithm comparison |
| 8 | Report Generator | `src/report_generator.py` | JSON/text reports, image export, comparison grid |
| 9 | Pipeline Controller | `src/pipeline.py` | Orchestrates all stages in sequence |
| 10 | CLI Entry Point | `main.py` | Argument parsing and command dispatch |

## User Workflow

```
Start
  ↓
Select input (image or video)
  ↓
Validate input file
  ↓
Load image / video
  ↓
Preprocess (grayscale, blur, histogram, Fourier)
  ↓
Extract features (edges, corners, descriptors)
  ↓
Segment / detect regions
  ↓
Analyse shapes (lines, circles, contours)
  ↓
[If video] Analyse motion (background, flow, parameters)
  ↓
Evaluate results (timing, metrics)
  ↓
Generate report (JSON, text, images)
  ↓
Save outputs
  ↓
End
```

## Testing Strategy

- **Unit tests** for preprocessing, feature extraction, and segmentation engines.
- All tests use **synthetic images** generated with NumPy — no external data dependencies.
- Test runner: `pytest tests/ -v`
