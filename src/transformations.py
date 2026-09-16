"""
Geometric Transformations — Module 1: Image Formation.

Implements Euclidean, affine, and projective transformations, plus
configurable resizing with interpolation control.
"""

import time
import cv2
import numpy as np


class GeometricTransformer:
    """
    Provides geometric transformation utilities for images.

    Supports Euclidean (rotation + translation), affine (parallelism preserved),
    and projective (full perspective) transformations.  Also provides resizing
    with selectable interpolation method.
    """

    INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def __init__(self):
        self.metadata = {}

    # ------------------------------------------------------------------
    # Euclidean transform  (rotation + translation — preserves distances)
    # ------------------------------------------------------------------

    def euclidean_transform(self, image: np.ndarray,
                            angle: float = 0.0,
                            tx: float = 0.0,
                            ty: float = 0.0) -> np.ndarray:
        """Apply a rigid (Euclidean) transformation.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        angle : float
            Rotation angle in **degrees** (counter-clockwise positive).
        tx, ty : float
            Translation in pixels (x-right, y-down).

        Returns
        -------
        np.ndarray
            Transformed image.
        """
        start = time.time()
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        rad = np.deg2rad(angle)

        # 2×3 Euclidean matrix: rotate around centre then translate
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + tx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + ty],
        ], dtype=np.float64)

        result = cv2.warpAffine(image, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT_101)
        self.metadata["euclidean_time"] = time.time() - start
        self.metadata["euclidean_params"] = {
            "angle": angle, "tx": tx, "ty": ty
        }
        return result

    # ------------------------------------------------------------------
    # Affine transform  (3 point correspondences — preserves parallelism)
    # ------------------------------------------------------------------

    def affine_transform(self, image: np.ndarray,
                         src_points: np.ndarray,
                         dst_points: np.ndarray) -> np.ndarray:
        """Apply an affine warp defined by three point correspondences.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        src_points : np.ndarray
            Source points, shape (3, 2), float32.
        dst_points : np.ndarray
            Destination points, shape (3, 2), float32.

        Returns
        -------
        np.ndarray
            Warped image.
        """
        start = time.time()
        h, w = image.shape[:2]
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        M = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT_101)
        self.metadata["affine_time"] = time.time() - start
        return result

    # ------------------------------------------------------------------
    # Projective transform  (4 point correspondences — full perspective)
    # ------------------------------------------------------------------

    def projective_transform(self, image: np.ndarray,
                             src_points: np.ndarray,
                             dst_points: np.ndarray) -> np.ndarray:
        """Apply a projective (homography) warp.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        src_points : np.ndarray
            Four source points, shape (4, 2), float32.
        dst_points : np.ndarray
            Four destination points, shape (4, 2), float32.

        Returns
        -------
        np.ndarray
            Warped image.
        """
        start = time.time()
        h, w = image.shape[:2]
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        M, _ = cv2.findHomography(src, dst)
        result = cv2.warpPerspective(image, M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT_101)
        self.metadata["projective_time"] = time.time() - start
        return result

    # ------------------------------------------------------------------
    # Resize with interpolation control
    # ------------------------------------------------------------------

    def resize(self, image: np.ndarray,
               width: int = None, height: int = None,
               scale: float = None,
               interpolation: str = "bilinear") -> np.ndarray:
        """Resize an image.

        Provide *either* explicit (width, height) *or* a uniform scale factor.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        width, height : int, optional
            Target dimensions.  Both must be given together.
        scale : float, optional
            Uniform scale factor (e.g. 0.5 for half-size).
        interpolation : str
            One of ``'nearest'``, ``'bilinear'``, ``'bicubic'``, ``'lanczos'``.

        Returns
        -------
        np.ndarray
            Resized image.
        """
        start = time.time()
        interp = self.INTERPOLATION_MAP.get(interpolation, cv2.INTER_LINEAR)

        if scale is not None:
            result = cv2.resize(image, None, fx=scale, fy=scale,
                                interpolation=interp)
        elif width is not None and height is not None:
            result = cv2.resize(image, (width, height), interpolation=interp)
        else:
            raise ValueError("Provide either (width, height) or scale.")

        self.metadata["resize_time"] = time.time() - start
        self.metadata["resize_output_shape"] = result.shape
        return result
