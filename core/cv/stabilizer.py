"""
====================================================================================================
DEFORMATION CORRECTION & GEOMETRIC STABILIZATION ENGINE
====================================================================================================

Physical Mechanics of Antique Punched Metal Music Disks:
---------------------------------------------------------
Historical music box disks (e.g., Polyphon, Symphonion, Kalliope, Regina, Stella) were stamped from
thin cold-rolled steel or zinc alloy sheet metal (0.3mm – 0.8mm gauge).

During playback in 19th/20th-century mechanical music boxes:
1. Pressure Rollers: Spring-loaded steel pressure rollers clamped the disk downward against the
   star-wheel gantry (reader plank) with several kilograms of force.
2. Plastic Deformation & Metal Creep: Over 100+ years of storage and operation under non-uniform
   mechanical stress, disks develop:
   - Conical Cupping (sagging in the middle or outer rim).
   - Asymmetric Radial Warping (uneven wavy borders).
   - Star-Wheel Reader Plank Bowing (one side bent downwards permanently).
3. Optical Perspective Distortion: When photographed, non-telecentric camera lenses and oblique
   shooting angles project concentric circular tracks into tilted, eccentric ellipses.

Mathematical Formulation of the Correction Pipeline:
-----------------------------------------------------
1. Global Elliptical Homography Rectification (Tilt & Conical Perspective):
   Fits an ellipse to the outer rim boundary points:
       (x' * cos(phi) + y' * sin(phi))^2 / a^2 + (-x' * sin(phi) + y' * cos(phi))^2 / b^2 = 1
   Computes eccentricity e = sqrt(1 - (b/a)^2) and transforms coordinates into an isotropic
   canonical circle where major axis 'a' equals minor axis 'b'.

2. Continuous Harmonic (Fourier) Boundary Normalization (Wobble & Plank Bend):
   The outer boundary radius R_rim(theta) as a continuous function of angle theta in [0, 2*pi)
   is decomposed into a Fourier series:
       R_rim(theta) = R_0 + sum_{k=1}^N [ a_k * cos(k * theta) + b_k * sin(k * theta) ]
   where:
   - k=1 models Center-of-Rotation Eccentricity (off-center arbor wobble).
   - k=2 models Elliptical Warping / Oblique Tilt leftover.
   - k=3, 4 models Saddle-shaped Bending & Reader Plank Clamp Deflection.

3. Radial Normalization of Musical Holes:
   For any detected hole at polar coordinate (theta_i, r_i), its normalized radius r_norm is:
       r_norm(theta_i, r_i) = r_i * [ R_target / R_rim(theta_i) ]
   This maps every hole back onto the ideal flat reference disk geometry with zero pitch drift.
====================================================================================================
"""

import math
import numpy as np
import numpy.linalg as la
import cv2
from typing import List, Tuple, Dict, Any, Optional


class PolarConverter:
    """
    Handles robust Cartesian <-> Polar transformations for disk hole coordinates.
    """

    @staticmethod
    def get_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two 2D points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Returns the angle in radians [0, pi] between 2D vectors v1 and v2."""
        cos_ang = np.dot(v1, v2)
        sin_ang = la.norm(np.cross(v1, v2))
        return np.arctan2(sin_ang, cos_ang)

    @staticmethod
    def quadrant(dot: Tuple[float, float]) -> int:
        """Determines the Cartesian quadrant of point relative to (0,0)."""
        x, y = dot
        if x > 0 and y >= 0:
            return 1
        elif x <= 0 and y > 0:
            return 2
        elif x < 0 and y <= 0:
            return 3
        elif x >= 0 and y < 0:
            return 4
        return 0

    @classmethod
    def to_polar(cls, dot: Tuple[float, float], center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
        """
        Converts centered (x,y) point to polar (theta, radius), where theta is in [0, 2*pi).
        """
        base = (1.0, 0.0)
        angle = cls.angle_between_vectors(base, dot)
        q = cls.quadrant(dot)
        if q > 2:
            angle = 2.0 * math.pi - angle
        
        radius = cls.get_distance(dot, center)
        return float(angle), float(radius)

    @classmethod
    def to_cartesian(cls, theta: float, radius: float, center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
        """Converts polar (theta, radius) back to Cartesian (x, y)."""
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        return float(x), float(y)


class EllipseRectifier:
    """
    Global Elliptical Homography Unwarper:
    Compensates for oblique camera perspective tilt and global conical cupping
    by fitting an ellipse to the disc boundary and rectifying it to a canonical circle.
    """

    @classmethod
    def fit_and_unwarp(
        cls, 
        boundary_contour: np.ndarray, 
        points: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], np.ndarray, Dict[str, float]]:
        """
        Fits a 5-parameter ellipse to boundary_contour and unwarps input points.
        
        Returns:
            - unwarped_points: Coordinates mapped to canonical circular geometry.
            - transform_matrix: 3x3 affine transformation matrix.
            - ellipse_params: Center, axes (a, b), angle (deg), and eccentricity.
        """
        if len(boundary_contour) < 5:
            # Not enough points for ellipse fit, return identity
            return points, np.eye(3), {"eccentricity": 0.0, "major_axis": 1.0, "minor_axis": 1.0}

        # OpenCV fitEllipse: returns ((center_x, center_y), (width, height), angle_deg)
        ellipse = cv2.fitEllipse(boundary_contour)
        (cx, cy), (width, height), angle_deg = ellipse

        major_axis = max(width, height) / 2.0
        minor_axis = min(width, height) / 2.0
        if major_axis == 0 or minor_axis == 0:
            return points, np.eye(3), {"eccentricity": 0.0, "major_axis": 1.0, "minor_axis": 1.0}

        eccentricity = math.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2))
        theta_rad = math.radians(angle_deg)

        # Transformation matrix to circularize ellipse:
        scale_factor = major_axis / minor_axis

        cos_t = math.cos(-theta_rad)
        sin_t = math.sin(-theta_rad)

        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        R_inv = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        S = np.array([[1.0, 0.0], [0.0, scale_factor]])

        M_2x2 = R_inv @ S @ R

        unwarped_points = []
        for x, y in points:
            centered = np.array([x - cx, y - cy])
            rectified = M_2x2 @ centered
            unwarped_points.append((float(rectified[0] + cx), float(rectified[1] + cy)))

        ellipse_params = {
            "center_x": float(cx),
            "center_y": float(cy),
            "major_axis": float(major_axis),
            "minor_axis": float(minor_axis),
            "angle_deg": float(angle_deg),
            "eccentricity": float(eccentricity),
            "aspect_ratio": float(scale_factor)
        }

        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
        A = np.eye(3)
        A[0:2, 0:2] = M_2x2
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=float)
        M_3x3 = T2 @ A @ T1

        return unwarped_points, M_3x3, ellipse_params


class HarmonicRimStabilizer:
    """
    Continuous Harmonic (Fourier Series) Boundary Deformation Modeler:
    ------------------------------------------------------------------
    Models continuous mechanical edge wobble, clamp sagging, and plastic cupping
    by fitting a low-order Fourier Series to the disk boundary contour R_rim(theta).
    
    This provides an analytical, smooth, and noise-immune continuous calibration
    function S(theta) = R_target / R_rim(theta) evaluated at any angle [0, 2*pi).
    """

    def __init__(self, harmonics: int = 4):
        """
        Args:
            harmonics (int): Number of Fourier harmonic terms (default: 4).
                             k=1 (eccentricity), k=2 (elliptical), k=3,4 (sagging/wobble).
        """
        self.harmonics = harmonics
        self.coeffs_a = []  # Cosine coefficients
        self.coeffs_b = []  # Sine coefficients
        self.r0 = 100.0     # Mean radius
        self.fitted = False

    def fit(self, boundary_contour: np.ndarray, center: Tuple[float, float]) -> float:
        """
        Fits Fourier coefficients to the disc's outer boundary contour relative to center.
        
        Returns:
            r0 (float): The mean reference radius of the disk.
        """
        if len(boundary_contour) < 10:
            self.fitted = False
            return self.r0

        # Sample boundary contour into polar (theta, radius)
        angles = []
        radii = []

        pts = boundary_contour.reshape(-1, 2)
        for pt in pts:
            rx = pt[0] - center[0]
            ry = pt[1] - center[1]
            th, r = PolarConverter.to_polar((rx, ry), (0, 0))
            angles.append(th)
            radii.append(r)

        angles = np.array(angles)
        radii = np.array(radii)

        # Construct Least-Squares Design Matrix:
        # [1, cos(1*th), sin(1*th), cos(2*th), sin(2*th), ..., cos(N*th), sin(N*th)]
        N = self.harmonics
        num_samples = len(angles)
        A = np.zeros((num_samples, 1 + 2 * N))
        A[:, 0] = 1.0

        for k in range(1, N + 1):
            A[:, 2 * k - 1] = np.cos(k * angles)
            A[:, 2 * k] = np.sin(k * angles)

        # Solve Normal Equations via SVD
        params, residuals, rank, s = la.lstsq(A, radii, rcond=None)

        self.r0 = float(params[0])
        self.coeffs_a = [float(params[2 * k - 1]) for k in range(1, N + 1)]
        self.coeffs_b = [float(params[2 * k]) for k in range(1, N + 1)]
        self.fitted = True

        return self.r0

    def evaluate_boundary_radius(self, theta: float) -> float:
        """
        Evaluates the analytical boundary radius at any continuous angle theta in radians.
        """
        if not self.fitted:
            return self.r0

        r = self.r0
        for k in range(1, self.harmonics + 1):
            ak = self.coeffs_a[k - 1]
            bk = self.coeffs_b[k - 1]
            r += ak * math.cos(k * theta) + bk * math.sin(k * theta)
        return max(1.0, float(r))

    def normalize_polar_hole(self, theta: float, raw_radius: float) -> float:
        """
        Applies local harmonic radial scaling factor to normalize a hole's radius.
        
        Formula:
            r_normalized = r_raw * (R_mean / R_boundary(theta))
        """
        local_boundary = self.evaluate_boundary_radius(theta)
        scale = self.r0 / local_boundary
        # Non-linear dampening: Center region is anchored, outer rim feels full deformation
        influence = raw_radius / self.r0
        calibrated_radius = raw_radius * (1.0 + (scale - 1.0) * influence)
        return float(calibrated_radius)


class DiskStabilizer:
    """
    High-level orchestrator for full geometric stabilization and deformation compensation:
    Combines Elliptical Homography unwarping, Harmonic Fourier boundary normalization,
    and fallback discrete tooth modeling.
    """

    @staticmethod
    def deduplicate_points(points: List[List[float]], threshold_distance: float = 10.0) -> List[List[float]]:
        """Removes duplicate or clustered center detections within threshold_distance."""
        if not points:
            return []
        
        arr = np.array(points)
        indices = np.lexsort((arr[:, 0], arr[:, 1]))
        sorted_data = arr[indices]
        
        result = [sorted_data[0].tolist()]
        prev = sorted_data[0]
        
        for i in range(1, len(sorted_data)):
            cur = sorted_data[i]
            if abs(cur[0] - prev[0]) > threshold_distance or abs(cur[1] - prev[1]) > threshold_distance:
                result.append(cur.tolist())
                prev = cur
                
        return result

    @staticmethod
    def classify_points_by_region(
        dedup_points: List[List[float]], 
        contours: List[np.ndarray], 
        center_perimeter: float, 
        outer_threshold_dist: float, 
        center: Tuple[float, float]
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """
        Categorizes points into:
        - melody_dots: Internal musical track holes
        - outer_dots: Outer fence/drive teeth reference markers
        - split_dots: Irregular / edge contour candidates
        """
        melody_dots = []
        outer_dots = []
        split_dots = []

        for pt in dedup_points:
            idx = int(pt[2])
            contour = contours[idx]
            perimeter = int(cv2.arcLength(contour, True))
            cx, cy = pt[0], pt[1]
            dist_to_center = PolarConverter.get_distance((cx, cy), center)
            
            val = [cx, cy, idx, dist_to_center]

            if center_perimeter > 0 and perimeter > (center_perimeter / 2.0):
                if perimeter < center_perimeter and perimeter > (center_perimeter / 2.0):
                    if dist_to_center > outer_threshold_dist:
                        outer_dots.append(val)
                    else:
                        split_dots.append(val)
                else:
                    split_dots.append(val)
            else:
                melody_dots.append(val)

        return melody_dots, outer_dots, split_dots

    @classmethod
    def stabilize_with_harmonic_model(
        cls,
        melody_dots: List[List[float]],
        outer_dots: List[List[float]],
        disc_boundary_contour: Optional[np.ndarray],
        center: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """
        Primary Stabilization Pipeline using Harmonic Fourier Model:
        1. Fits continuous Fourier series to outer boundary contour.
        2. Applies smooth continuous radial correction to all melody holes.
        3. Returns calibrated coordinates and deformation diagnostics.
        """
        stabilizer = HarmonicRimStabilizer(harmonics=4)
        
        # Determine boundary source: either continuous perimeter contour or outer tooth centers
        if disc_boundary_contour is not None and len(disc_boundary_contour) >= 10:
            mean_radius = stabilizer.fit(disc_boundary_contour, center)
        elif len(outer_dots) >= 5:
            outer_pts = np.array([[d[0], d[1]] for d in outer_dots], dtype=np.int32).reshape(-1, 1, 2)
            mean_radius = stabilizer.fit(outer_pts, center)
        else:
            raw_radii = [PolarConverter.get_distance((d[0], d[1]), center) for d in melody_dots]
            mean_radius = float(np.max(raw_radii)) if raw_radii else 100.0

        # Calibrate melody holes using continuous harmonic equation
        upd_main = []
        for dot in melody_dots:
            rx = dot[0] - center[0]
            ry = dot[1] - center[1]
            th, raw_r = PolarConverter.to_polar((rx, ry), (0, 0))
            calibrated_r = stabilizer.normalize_polar_hole(th, raw_r)
            upd_main.append([th, calibrated_r])

        # Calibrate outer teeth for visualization/inspection
        upd_top = []
        for dot in outer_dots:
            rx = dot[0] - center[0]
            ry = dot[1] - center[1]
            th, raw_r = PolarConverter.to_polar((rx, ry), (0, 0))
            calibrated_r = stabilizer.normalize_polar_hole(th, raw_r)
            upd_top.append([th, calibrated_r])

        upd_main_arr = np.array(upd_main) if upd_main else np.empty((0, 2))
        upd_top_arr = np.array(upd_top) if upd_top else np.empty((0, 2))

        if len(upd_main_arr) > 0:
            upd_main_arr = upd_main_arr[np.argsort(upd_main_arr[:, 0])]

        diagnostics = {
            "mean_radius_px": float(mean_radius),
            "harmonic_model_fitted": stabilizer.fitted,
            "fourier_coeffs_cosine": stabilizer.coeffs_a,
            "fourier_coeffs_sine": stabilizer.coeffs_b,
            "max_radial_wobble_px": float(max(stabilizer.coeffs_a + stabilizer.coeffs_b, default=0.0))
        }

        return upd_top_arr, upd_main_arr, mean_radius, diagnostics
