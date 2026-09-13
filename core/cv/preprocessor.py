import math
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional


class ImagePreprocessor:
    """Handles image loading, contour extraction, center detection, and cropping."""

    @staticmethod
    def load_grayscale(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image file at {image_path}")
        return img

    @staticmethod
    def load_color(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image file at {image_path}")
        return img

    @staticmethod
    def get_main_contours(contours: List[np.ndarray], top_n: int = 3) -> List[np.ndarray]:
        """Sort contours by perimeter and return the largest ones."""
        sorted_contours = []
        for i, cnt in enumerate(contours):
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                sorted_contours.append((perimeter, i))
        
        if not sorted_contours:
            return []

        sorted_contours.sort(key=lambda x: x[0], reverse=True)
        best_cnts = [contours[idx] for _, idx in sorted_contours[:top_n]]
        return best_cnts

    @staticmethod
    def crop_contour(img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop image to the bounding box of the given contour."""
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        y_indices, x_indices = np.where(mask == 255)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return img.copy(), (0, 0)
        
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        cropped = img[min_y:max_y + 1, min_x:max_x + 1]
        return cropped, (min_x, min_y)

    @classmethod
    def process_contours(
        cls, 
        img: np.ndarray, 
        disk_radius_cm: float = 19.0
    ) -> Dict[str, Any]:
        """
        Processes disc image on the FULL uncropped frame so no parts of the disk or holes are cut off.
        """
        height, width = img.shape[:2]
        pred_center = [width / 2.0, height / 2.0]
        min_center_radius = (1.5 / disk_radius_cm) * (min(width, height) / 2.0)

        # 1. Edge detection on full image (adaptive filtering)
        bilateral = cv2.bilateralFilter(img, 5, 175, 175)
        edges = cv2.Canny(bilateral, 60, 180)
        hole_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Main disk outline contour (for deformation modeling)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours_main, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        main_cnts = cls.get_main_contours(contours_main, top_n=3)
        target_cnt = main_cnts[0] if main_cnts else np.array([])
        disk_perimeter = int(cv2.arcLength(target_cnt, True)) if len(target_cnt) > 0 else 0

        per_list = []
        center = [int(pred_center[0]), int(pred_center[1])]
        center_perimeter = 0
        seen_points = {}

        for i, contour in enumerate(hole_contours):
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            perimeter = int(cv2.arcLength(contour, True))
            
            if len(approx) > 1 and perimeter > 1 and area > 1:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    key = f"{cX}_{cY}"
                    if key not in seen_points:
                        seen_points[key] = (cX, cY)
                        per_list.append([cX, cY, i])
                        
                        # Check if this contour is near the estimated arbor center
                        if abs(height / 2.0 - cY) <= min_center_radius and abs(width / 2.0 - cX) <= min_center_radius:
                            center = [cX, cY]
                            center_perimeter = perimeter

        outer_radius_px = min(width, height) / 2.0 * 0.95

        return {
            "cropped_img": img,  # Full uncropped frame
            "contours": hole_contours,
            "boundary_contour": target_cnt,
            "points_list": per_list,
            "center": center,
            "center_perimeter": center_perimeter,
            "disk_perimeter": disk_perimeter,
            "outer_radius_px": outer_radius_px,
            "pred_center": pred_center,
            "offset": (0, 0)
        }
