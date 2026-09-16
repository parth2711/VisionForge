"""
Pattern & Shape Analysis Engine — Module 4: Pattern Analysis.

Implements Hough line/circle detection, contour analysis with shape
descriptors, PCA dimensionality reduction, and KNN classification.
"""

import time
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PatternAnalyzer:
    """
    Pattern analysis and shape detection engine.

    Provides Hough Transform-based line and circle detection, contour
    analysis with geometric properties, PCA for feature-space reduction,
    and a simple KNN classifier for demonstration purposes.
    """

    def __init__(self):
        self.metadata = {}

    # ------------------------------------------------------------------
    # Hough line detection
    # ------------------------------------------------------------------

    def hough_lines(self, edge_image: np.ndarray,
                    original_image: np.ndarray = None,
                    rho: float = 1.0,
                    theta_resolution: float = None,
                    threshold: int = 80,
                    min_length: int = 50,
                    max_gap: int = 10) -> dict:
        """Detect lines using the probabilistic Hough Transform.

        Parameters
        ----------
        edge_image : np.ndarray
            Binary edge map (e.g. from Canny).
        original_image : np.ndarray, optional
            BGR image on which to draw detected lines.
        rho : float
            Distance resolution of the accumulator (pixels).
        theta_resolution : float, optional
            Angle resolution (radians).  Defaults to π/180.
        threshold : int
            Accumulator threshold.
        min_length : int
            Minimum line length.
        max_gap : int
            Maximum gap between line segments.

        Returns
        -------
        dict
            ``{"annotated_image": np.ndarray, "lines": list,
               "line_count": int, "line_stats": dict}``
        """
        start = time.time()
        if theta_resolution is None:
            theta_resolution = np.pi / 180

        if original_image is not None:
            vis = original_image.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        else:
            vis = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(
            edge_image, rho, theta_resolution, threshold,
            minLineLength=min_length, maxLineGap=max_gap,
        )

        line_list = []
        lengths = []
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                line_list.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "length": length, "angle": angle,
                })
                lengths.append(length)
                angles.append(angle)

        stats = {}
        if lengths:
            stats = {
                "mean_length": float(np.mean(lengths)),
                "max_length": float(np.max(lengths)),
                "mean_angle": float(np.mean(angles)),
                "dominant_orientation": "horizontal"
                if abs(np.mean(angles)) < 45 else "vertical",
            }

        result = {
            "annotated_image": vis,
            "lines": line_list,
            "line_count": len(line_list),
            "line_stats": stats,
        }

        self.metadata["hough_lines_time"] = time.time() - start
        self.metadata["hough_line_count"] = len(line_list)
        return result

    # ------------------------------------------------------------------
    # Hough circle detection
    # ------------------------------------------------------------------

    def hough_circles(self, image: np.ndarray,
                      dp: float = 1.2,
                      min_dist: int = 50,
                      param1: int = 100,
                      param2: int = 30,
                      min_radius: int = 10,
                      max_radius: int = 200) -> dict:
        """Detect circles using the Hough Circle Transform.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale).
        dp : float
            Inverse ratio of accumulator resolution.
        min_dist : int
            Minimum distance between circle centres.
        param1, param2 : int
            Canny and accumulator threshold parameters.
        min_radius, max_radius : int
            Radius constraints.

        Returns
        -------
        dict
            ``{"annotated_image": np.ndarray, "circles": list,
               "circle_count": int}``
        """
        start = time.time()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vis = image.copy()
        else:
            gray = image
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp, min_dist,
            param1=param1, param2=param2,
            minRadius=min_radius, maxRadius=max_radius,
        )

        circle_list = []
        if circles is not None:
            circles_int = np.uint16(np.around(circles))
            for c in circles_int[0, :]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 2, (0, 0, 255), 3)
                circle_list.append({
                    "center_x": cx, "center_y": cy,
                    "radius": r, "area": float(np.pi * r * r),
                })

        result = {
            "annotated_image": vis,
            "circles": circle_list,
            "circle_count": len(circle_list),
        }

        self.metadata["hough_circles_time"] = time.time() - start
        self.metadata["hough_circle_count"] = len(circle_list)
        return result

    # ------------------------------------------------------------------
    # Contour analysis
    # ------------------------------------------------------------------

    def find_contours(self, image: np.ndarray) -> dict:
        """Detect contours and compute shape descriptors.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale).

        Returns
        -------
        dict
            ``{"contour_image": np.ndarray, "contours": list,
               "hierarchy": np.ndarray, "properties": list,
               "contour_count": int}``
        """
        start = time.time()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vis = image.copy()
        else:
            gray = image
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Draw contours
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        # Compute properties for significant contours
        properties = self.contour_properties(contours)

        result = {
            "contour_image": vis,
            "contours": contours,
            "hierarchy": hierarchy,
            "properties": properties,
            "contour_count": len(contours),
        }

        self.metadata["contour_time"] = time.time() - start
        self.metadata["contour_count"] = len(contours)
        return result

    def contour_properties(self, contours: list,
                           min_area: float = 100) -> list:
        """Compute geometric properties for each contour.

        Parameters
        ----------
        contours : list
            List of OpenCV contours.
        min_area : float
            Minimum area to consider a contour significant.

        Returns
        -------
        list[dict]
            Per-contour properties.
        """
        props = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            circularity = (4 * np.pi * area / (perimeter ** 2)
                           if perimeter > 0 else 0)

            props.append({
                "area": float(area),
                "perimeter": float(perimeter),
                "bounding_rect": {"x": int(x), "y": int(y),
                                  "w": int(w), "h": int(h)},
                "aspect_ratio": float(aspect_ratio),
                "extent": float(extent),
                "solidity": float(solidity),
                "circularity": float(circularity),
            })
        return props

    # ------------------------------------------------------------------
    # PCA analysis
    # ------------------------------------------------------------------

    def pca_analysis(self, feature_vectors: np.ndarray,
                     n_components: int = 2) -> dict:
        """Perform PCA dimensionality reduction on feature vectors.

        Parameters
        ----------
        feature_vectors : np.ndarray
            2-D array (num_samples × num_features).
        n_components : int
            Number of principal components to retain.

        Returns
        -------
        dict
            ``{"transformed": np.ndarray,
               "explained_variance_ratio": list,
               "total_variance_explained": float,
               "components": np.ndarray}``
        """
        start = time.time()
        pca = PCA(n_components=min(n_components, feature_vectors.shape[1]))
        transformed = pca.fit_transform(feature_vectors)

        result = {
            "transformed": transformed,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(
                sum(pca.explained_variance_ratio_)),
            "components": pca.components_,
        }

        self.metadata["pca_time"] = time.time() - start
        self.metadata["pca_variance_explained"] = result[
            "total_variance_explained"]
        return result

    # ------------------------------------------------------------------
    # KNN classification (demo)
    # ------------------------------------------------------------------

    def knn_classify(self, train_features: np.ndarray,
                     train_labels: np.ndarray,
                     test_features: np.ndarray,
                     k: int = 3) -> dict:
        """Simple K-Nearest-Neighbours classification.

        Parameters
        ----------
        train_features : np.ndarray
            Training feature matrix (n_train × d).
        train_labels : np.ndarray
            Training labels (n_train,).
        test_features : np.ndarray
            Test feature matrix (n_test × d).
        k : int
            Number of neighbours.

        Returns
        -------
        dict
            ``{"predictions": np.ndarray, "k": int}``
        """
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)
        predictions = knn.predict(test_features)

        result = {
            "predictions": predictions,
            "k": k,
        }

        self.metadata["knn_time"] = time.time() - start
        return result

    # ------------------------------------------------------------------
    # Full shape analysis pipeline
    # ------------------------------------------------------------------

    def run_shape_analysis(self, image: np.ndarray,
                           edge_image: np.ndarray = None,
                           output_dir: str = None) -> dict:
        """Run Hough detection + contour analysis on an image.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.
        edge_image : np.ndarray, optional
            Pre-computed binary edge map.  If not provided, Canny is
            applied internally.
        output_dir : str, optional
            Directory for saving output images.

        Returns
        -------
        dict
            Combined results from Hough lines, circles, and contours.
        """
        if edge_image is None:
            gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if len(image.shape) == 3 else image)
            edge_image = cv2.Canny(gray, 50, 150)

        results = {}
        results["hough_lines"] = self.hough_lines(edge_image, image)
        results["hough_circles"] = self.hough_circles(image)
        results["contours"] = self.find_contours(image)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "hough_lines.png"),
                        results["hough_lines"]["annotated_image"])
            cv2.imwrite(os.path.join(output_dir, "hough_circles.png"),
                        results["hough_circles"]["annotated_image"])
            cv2.imwrite(os.path.join(output_dir, "contours.png"),
                        results["contours"]["contour_image"])

        results["metadata"] = self.metadata.copy()
        return results
