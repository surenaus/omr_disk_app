import cv2
import numpy as np
from typing import List, Tuple, Any, Dict


class DetectionVisualizer:
    """Renders visual debug overlays onto the disk image for user inspection."""

    @staticmethod
    def draw_detections(
        cropped_img: np.ndarray,
        center: Tuple[int, int],
        melody_dots: List[List[float]],
        outer_dots: List[List[float]],
        split_dots: List[List[float]],
        output_path: str
    ) -> str:
        """
        Draws colored overlays:
        - Center: Large cyan ring & dot
        - Melody Holes: Green dots
        - Outer Rim Teeth: Red dots
        - Edge / Split contours: Orange dots
        """
        # Convert to 3-channel BGR if grayscale
        if len(cropped_img.shape) == 2:
            vis_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = cropped_img.copy()

        # Draw Center
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(vis_img, (cx, cy), 12, (255, 255, 0), 2)  # Cyan circle
        cv2.circle(vis_img, (cx, cy), 3, (0, 0, 255), -1)   # Red center dot
        cv2.putText(vis_img, "Center", (cx + 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw Outer Drive Teeth (Red)
        for dot in outer_dots:
            x, y = int(dot[0]), int(dot[1])
            cv2.circle(vis_img, (x, y), 5, (0, 0, 255), -1)

        # Draw Irregular / Split Dots (Orange)
        for dot in split_dots:
            x, y = int(dot[0]), int(dot[1])
            cv2.circle(vis_img, (x, y), 4, (0, 165, 255), -1)

        # Draw Musical Holes (Bright Green)
        for dot in melody_dots:
            x, y = int(dot[0]), int(dot[1])
            cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)

        cv2.imwrite(output_path, vis_img)
        return output_path
