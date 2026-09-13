"""
====================================================================================================
ADAPTIVE RADIAL DENSITY QUANTIZER & PHYSICAL COMB CALIBRATOR
====================================================================================================

Physical Architecture of the 78-Note Music Box Comb & Star-Wheel Gantry:
-------------------------------------------------------------------------
Historical disk music boxes (e.g. Polyphon 19.5" or Symphonion) produce sound via a steel comb:
1. Dual / Triple Star-Wheel Gantry:
   Above the steel comb, a fixed reader gantry holds 78 independent star-wheels (spur pinwheels).
   When a hole with an underlying punched metal plectrum passes beneath a star-wheel, it rotates
   the wheel by 1 tooth (1/4 to 1/6 turn), which plucks the corresponding comb tooth.

2. Physical Register Layout (78 Tracks):
   - Group 1 (Inner Tracks 0-24, 25 notes): Bass & Tenor registers (Pitches 44-68).
   - Margin 1 (Track 25): Physical structural spacer / bearing dead-zone (0 notes).
   - Group 2 (Middle Tracks 26-50, 25 notes): Alto & Soprano registers (Pitches 69-85 with octave doubles).
   - Margin 2 (Track 51): Physical structural spacer / bearing dead-zone (0 notes).
   - Group 3 (Outer Tracks 52-77, 26 notes): High Soprano & Harmonic Bells (Pitches 85-103).

3. The Deformation Problem in Fixed-Grid Binning:
   Because the disk metal bent under the reader plank over decades, the true physical distance
   between the inner track (r_low) and outer track (r_high) varies non-linearly.
   A rigid linear grid (step = (high - low)/78) causes holes in slightly stretched or compressed
   annular zones to fall into the wrong adjacent row (octave or semitone pitch corruption).

4. Solution: Adaptive Radial Density Peak Alignment:
   - Compute a 1D Kernel Density Estimate (KDE) or fine-binned histogram of all hole radii:
         D(r) = sum_{i=1}^M K_h(r - r_i)
   - Locate prominent local maxima (peaks) in D(r), which correspond directly to the physical
     concentric tracks where multiple musical notes were punched around the 360-degree cycle.
   - Snap hole radii to the nearest calibrated comb track ridge.
====================================================================================================
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class TrackQuantizer:
    """
    Quantizes stabilized polar holes into physical note tracks (default: 78 tracks).
    Uses Adaptive Radial Density Peak Alignment and cleans up duplicate / overlapping punches.
    """

    DEFAULT_PITCH_MAP = [
        # Group 1 (Rows 0-24) -> Pitches 44 to 68 (Bass to Lower Mid)
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
        # Margin 1 (Row 25) -> Non-playing structural separator
        0,
        # Group 2 (Rows 26-50) -> Pitches 69 to 85 (Mid to Upper Mid with unison reeds)
        69, 70, 70, 71, 72, 73, 74, 75, 76, 77, 77, 78, 78, 78, 79, 80, 80, 81, 82, 82, 82, 83, 83, 84, 85,
        # Margin 2 (Row 51) -> Non-playing structural separator
        0,
        # Group 3 (Rows 52-77) -> Pitches 85 to 103 (Treble & High Chimes)
        85, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 92, 93, 94, 94, 95, 95, 96, 97, 97, 98, 99, 100, 101, 102, 103
    ]

    @classmethod
    def compute_theoretical_track_centers(
        cls,
        mean_line_radius: float,
        disk_radius_cm: float = 19.0,
        center_to_sound_cm: float = 2.45,
        sound_to_end_cm: float = 0.85,
        playing_span_cm: float = 15.7,
        row_count: int = 78
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Calculates theoretical physical track centers and boundaries based on disc geometry.
        """
        total_sectors = 26 * 3 * 2 + 1  # Standard physical grid subdivision
        low_real = (center_to_sound_cm / disk_radius_cm) * mean_line_radius
        one_point = ((playing_span_cm / total_sectors) / disk_radius_cm) * mean_line_radius
        high_part = (sound_to_end_cm / disk_radius_cm) * mean_line_radius
        high = mean_line_radius - high_part
        
        left_exclude = low_real + (one_point * 0.5)
        right_exclude = high - (0.5 * one_point)
        step = (right_exclude - left_exclude) / float(row_count)

        track_centers = np.array([left_exclude + (i + 0.5) * step for i in range(row_count)])
        return track_centers, left_exclude, right_exclude, step

    @classmethod
    def find_density_peaks(cls, radii: np.ndarray, num_bins: int = 300) -> np.ndarray:
        """
        Estimates radial density distribution and finds local cluster peaks.
        """
        if len(radii) < 10:
            return np.array([])

        counts, bin_edges = np.histogram(radii, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Simple 3-point moving average smoothing
        smoothed = np.convolve(counts, np.ones(3)/3.0, mode='same')

        # Local maxima detection
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1] and smoothed[i] >= 2:
                peaks.append(bin_centers[i])

        return np.array(peaks)

    @classmethod
    def bin_holes_to_tracks(
        cls,
        stabilized_dots: np.ndarray,
        mean_line_radius: float,
        disk_radius_cm: float = 19.0,
        center_to_sound_cm: float = 2.45,
        sound_to_end_cm: float = 0.85,
        playing_span_cm: float = 15.7,
        row_count: int = 78,
        use_density_calibration: bool = True
    ) -> List[List[float]]:
        """
        Assigns stabilized holes into 78 discrete note tracks.
        Uses adaptive density peak alignment to calibrate track boundaries against physical warping.
        """
        if len(stabilized_dots) == 0 or mean_line_radius <= 0:
            return [[] for _ in range(row_count)]

        track_centers, left_exclude, right_exclude, step = cls.compute_theoretical_track_centers(
            mean_line_radius=mean_line_radius,
            disk_radius_cm=disk_radius_cm,
            center_to_sound_cm=center_to_sound_cm,
            sound_to_end_cm=sound_to_end_cm,
            playing_span_cm=playing_span_cm,
            row_count=row_count
        )

        all_radii = stabilized_dots[:, 1]

        # Optional Adaptive Density Peak Calibration:
        # If density peaks are found, fit a subtle affine warp to adjust track center alignment
        calibrated_centers = track_centers.copy()
        if use_density_calibration and len(all_radii) >= 50:
            peaks = cls.find_density_peaks(all_radii)
            if len(peaks) >= 15:
                # Find best matching theoretical track center for each empirical peak
                matched_theo = []
                matched_peak = []
                for p in peaks:
                    diffs = np.abs(track_centers - p)
                    nearest_idx = np.argmin(diffs)
                    if diffs[nearest_idx] < step * 0.75:
                        matched_theo.append(track_centers[nearest_idx])
                        matched_peak.append(p)
                
                if len(matched_theo) >= 10:
                    # Fit 1st-degree polynomial: peak = scale * theoretical + shift
                    poly = np.polyfit(matched_theo, matched_peak, 1)
                    calibrated_centers = np.polyval(poly, track_centers)

        # Build nearest-center quantization intervals
        punched_card = [[] for _ in range(row_count)]
        
        for dot in stabilized_dots:
            theta = float(dot[0])
            radius = float(dot[1])

            # Check if radius is within valid playing area
            if radius < (left_exclude - step) or radius > (right_exclude + step):
                continue

            # Find nearest track center
            dist_to_centers = np.abs(calibrated_centers - radius)
            best_track_idx = int(np.argmin(dist_to_centers))

            if dist_to_centers[best_track_idx] <= (step * 0.65):
                punched_card[best_track_idx].append(theta)

        return punched_card

    @staticmethod
    def filter_close_punches(
        punched_card: List[List[float]], 
        min_angular_distance: float = 0.05
    ) -> List[List[float]]:
        """
        Removes overlapping or duplicate hole detections that are physically too close
        (e.g., star-wheel needs minimum mechanical recovery angle to reset plectrum).
        """
        cleaned_card = []
        for row in punched_card:
            if len(row) <= 1:
                cleaned_card.append(row)
                continue
            
            sorted_angles = sorted(row)
            filtered_row = [sorted_angles[0]]
            for angle in sorted_angles[1:]:
                if abs(angle - filtered_row[-1]) >= min_angular_distance:
                    filtered_row.append(angle)
            cleaned_card.append(filtered_row)
            
        return cleaned_card
