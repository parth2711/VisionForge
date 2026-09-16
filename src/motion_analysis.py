"""
Motion Analysis Engine — Module 4: Motion & Video Analysis.

Implements background subtraction (MOG2 / KNN), dense optical flow
(Farneback), sparse optical flow (Lucas-Kanade / KLT), and motion
parameter estimation from flow fields.
"""

import time
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MotionAnalyzer:
    """
    Video motion analysis engine.

    Provides background subtraction, dense and sparse optical flow, and
    motion parameter estimation.  All methods operate on video files
    and produce annotated output frames and statistics.
    """

    def __init__(self):
        self.metadata = {}

    # ------------------------------------------------------------------
    # Helper: read video frames
    # ------------------------------------------------------------------

    @staticmethod
    def _read_frames(video_path: str, max_frames: int = 300) -> list:
        """Read up to *max_frames* frames from a video file.

        Returns
        -------
        list[np.ndarray]
            List of BGR frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video: {video_path}")
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if len(frames) < 2:
            raise ValueError(
                "Video must contain at least 2 frames for motion analysis.")
        return frames

    # ------------------------------------------------------------------
    # Background subtraction
    # ------------------------------------------------------------------

    def background_subtraction(self, video_path: str,
                               method: str = "mog2",
                               learning_rate: float = -1,
                               max_frames: int = 300) -> dict:
        """Compute a background model and foreground masks.

        Parameters
        ----------
        video_path : str
            Path to the input video file.
        method : str
            ``'mog2'`` for ``BackgroundSubtractorMOG2``,
            ``'knn'`` for ``BackgroundSubtractorKNN``.
        learning_rate : float
            OpenCV learning rate (−1 for automatic).
        max_frames : int
            Maximum frames to process.

        Returns
        -------
        dict
            ``{"foreground_masks": list, "background_model": np.ndarray,
               "sample_foreground": np.ndarray, "stats": dict}``
        """
        start = time.time()
        frames = self._read_frames(video_path, max_frames)

        if method == "mog2":
            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=len(frames), varThreshold=16,
                detectShadows=True,
            )
        elif method == "knn":
            subtractor = cv2.createBackgroundSubtractorKNN(
                history=len(frames), detectShadows=True,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'mog2' or 'knn'.")

        fg_masks = []
        for frame in frames:
            fg = subtractor.apply(frame, learningRate=learning_rate)
            fg_masks.append(fg)

        background = subtractor.getBackgroundImage()

        # Statistics
        avg_fg_pixels = float(np.mean([np.count_nonzero(m) for m in fg_masks]))
        total_pixels = frames[0].shape[0] * frames[0].shape[1]

        result = {
            "foreground_masks": fg_masks,
            "background_model": background,
            "sample_foreground": fg_masks[len(fg_masks) // 2],
            "stats": {
                "frames_processed": len(frames),
                "avg_foreground_pixels": avg_fg_pixels,
                "avg_foreground_ratio": avg_fg_pixels / total_pixels,
                "method": method,
            },
        }

        self.metadata["bg_sub_time"] = time.time() - start
        self.metadata["bg_sub_method"] = method
        self.metadata["bg_sub_frames"] = len(frames)
        return result

    # ------------------------------------------------------------------
    # Dense optical flow (Farneback)
    # ------------------------------------------------------------------

    def optical_flow_dense(self, video_path: str,
                           max_frames: int = 300) -> dict:
        """Compute dense optical flow using Farneback's algorithm.

        Parameters
        ----------
        video_path : str
            Path to the input video.
        max_frames : int
            Maximum frames to process.

        Returns
        -------
        dict
            ``{"flow_visualizations": list, "magnitude_stats": dict,
               "sample_flow": np.ndarray}``
        """
        start = time.time()
        frames = self._read_frames(video_path, max_frames)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        flow_vis_list = []
        magnitudes = []

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )

            # HSV visualisation
            mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
            hsv = np.zeros((*prev_gray.shape, 3), dtype=np.uint8)
            hsv[:, :, 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
            hsv[:, :, 1] = 255
            hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255,
                                          cv2.NORM_MINMAX).astype(np.uint8)
            bgr_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow_vis_list.append(bgr_vis)
            magnitudes.append(float(np.mean(mag)))

            prev_gray = curr_gray

        result = {
            "flow_visualizations": flow_vis_list,
            "sample_flow": flow_vis_list[len(flow_vis_list) // 2],
            "magnitude_stats": {
                "mean_magnitude": float(np.mean(magnitudes)),
                "max_magnitude": float(np.max(magnitudes)),
                "min_magnitude": float(np.min(magnitudes)),
                "frames_processed": len(magnitudes),
            },
        }

        self.metadata["dense_flow_time"] = time.time() - start
        self.metadata["dense_flow_mean_mag"] = result[
            "magnitude_stats"]["mean_magnitude"]
        return result

    # ------------------------------------------------------------------
    # Sparse optical flow (Lucas-Kanade / KLT)
    # ------------------------------------------------------------------

    def optical_flow_sparse(self, video_path: str,
                            max_corners: int = 100,
                            max_frames: int = 300) -> dict:
        """Track sparse feature points using Lucas-Kanade (KLT).

        Parameters
        ----------
        video_path : str
            Path to the input video.
        max_corners : int
            Maximum number of feature points to track.
        max_frames : int
            Maximum frames to process.

        Returns
        -------
        dict
            ``{"tracked_image": np.ndarray, "trajectories": list,
               "stats": dict}``
        """
        start = time.time()
        frames = self._read_frames(video_path, max_frames)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        # Shi-Tomasi corners
        feature_params = dict(
            maxCorners=max_corners, qualityLevel=0.3,
            minDistance=7, blockSize=7,
        )
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is None:
            return {
                "tracked_image": frames[0],
                "trajectories": [],
                "stats": {"points_tracked": 0},
            }

        # Lucas-Kanade parameters
        lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      10, 0.03),
        )

        # Create colour mask for drawing trajectories
        mask = np.zeros_like(frames[0])
        colors = np.random.randint(0, 255, (max_corners, 3)).tolist()

        trajectories = [[] for _ in range(len(p0))]
        for pt_idx, pt in enumerate(p0):
            trajectories[pt_idx].append(tuple(pt.ravel().tolist()))

        vis = frames[-1].copy()
        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None, **lk_params,
            )

            if p1 is not None:
                good_new = p1[status.flatten() == 1]
                good_old = p0[status.flatten() == 1]

                active = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    color = colors[i % len(colors)]
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                                   color, 2)
                    vis = cv2.circle(frame.copy(), (int(a), int(b)), 5,
                                    color, -1)
                    active.append(i)

                p0 = good_new.reshape(-1, 1, 2)
            prev_gray = curr_gray

        vis = cv2.add(vis, mask)

        # Serialisable trajectories (just summary)
        traj_summary = []
        for t in trajectories:
            if len(t) > 0:
                traj_summary.append({
                    "start": t[0],
                    "length": len(t),
                })

        result = {
            "tracked_image": vis,
            "trajectories": traj_summary,
            "stats": {
                "initial_points": len(trajectories),
                "frames_processed": len(frames) - 1,
            },
        }

        self.metadata["sparse_flow_time"] = time.time() - start
        self.metadata["sparse_flow_points"] = len(trajectories)
        return result

    # ------------------------------------------------------------------
    # Motion parameter estimation
    # ------------------------------------------------------------------

    def estimate_motion_parameters(self, video_path: str,
                                   max_frames: int = 100) -> dict:
        """Estimate global motion parameters from optical flow.

        Fits an affine model between consecutive frames and extracts
        translation, rotation, and scale.

        Parameters
        ----------
        video_path : str
            Path to the input video.
        max_frames : int
            Maximum frames to process.

        Returns
        -------
        dict
            ``{"translations": list, "rotations": list,
               "scales": list, "summary": dict}``
        """
        start = time.time()
        frames = self._read_frames(video_path, max_frames)

        translations = []
        rotations = []
        scales = []

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect features
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01,
                minDistance=30, blockSize=3,
            )
            if prev_pts is None:
                prev_gray = curr_gray
                continue

            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None,
            )

            if curr_pts is None:
                prev_gray = curr_gray
                continue

            good_prev = prev_pts[status.flatten() == 1]
            good_curr = curr_pts[status.flatten() == 1]

            if len(good_prev) < 3:
                prev_gray = curr_gray
                continue

            # Estimate affine transform
            M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
            if M is not None:
                tx = float(M[0, 2])
                ty = float(M[1, 2])
                sx = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
                rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

                translations.append({"tx": tx, "ty": ty})
                rotations.append(rotation)
                scales.append(sx)

            prev_gray = curr_gray

        summary = {}
        if translations:
            tx_vals = [t["tx"] for t in translations]
            ty_vals = [t["ty"] for t in translations]
            summary = {
                "mean_translation_x": float(np.mean(tx_vals)),
                "mean_translation_y": float(np.mean(ty_vals)),
                "mean_rotation_deg": float(np.mean(rotations)),
                "mean_scale": float(np.mean(scales)),
                "total_displacement": float(
                    np.sqrt(sum(t["tx"] ** 2 + t["ty"] ** 2
                                for t in translations))),
                "frames_analysed": len(translations),
            }

        result = {
            "translations": translations,
            "rotations": rotations,
            "scales": scales,
            "summary": summary,
        }

        self.metadata["motion_params_time"] = time.time() - start
        self.metadata["motion_summary"] = summary
        return result

    # ------------------------------------------------------------------
    # Full motion analysis pipeline
    # ------------------------------------------------------------------

    def run_full_motion_analysis(self, video_path: str,
                                 output_dir: str = None) -> dict:
        """Run background subtraction + optical flow + motion estimation.

        Parameters
        ----------
        video_path : str
            Path to the input video.
        output_dir : str, optional
            Directory for saving output images.

        Returns
        -------
        dict
            Combined results from all motion analysis methods.
        """
        results = {}

        # Background subtraction
        bg_result = self.background_subtraction(video_path)
        results["background_subtraction"] = {
            "stats": bg_result["stats"],
        }
        bg_model = bg_result.get("background_model")
        sample_fg = bg_result.get("sample_foreground")

        # Dense optical flow
        dense_result = self.optical_flow_dense(video_path)
        results["dense_optical_flow"] = {
            "magnitude_stats": dense_result["magnitude_stats"],
        }
        sample_flow = dense_result.get("sample_flow")

        # Sparse optical flow (KLT)
        sparse_result = self.optical_flow_sparse(video_path)
        results["sparse_optical_flow"] = {
            "stats": sparse_result["stats"],
        }
        tracked_img = sparse_result.get("tracked_image")

        # Motion parameters
        motion_result = self.estimate_motion_parameters(video_path)
        results["motion_parameters"] = {
            "summary": motion_result["summary"],
        }

        # Save outputs
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if bg_model is not None:
                cv2.imwrite(os.path.join(output_dir,
                                         "background_model.png"), bg_model)
            if sample_fg is not None:
                cv2.imwrite(os.path.join(output_dir,
                                         "foreground_mask.png"), sample_fg)
            if sample_flow is not None:
                cv2.imwrite(os.path.join(output_dir,
                                         "optical_flow_dense.png"),
                            sample_flow)
            if tracked_img is not None:
                cv2.imwrite(os.path.join(output_dir,
                                         "optical_flow_sparse.png"),
                            tracked_img)

        results["metadata"] = self.metadata.copy()
        return results
