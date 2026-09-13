import os
import uuid
import numpy as np
from typing import Dict, Any, Optional, List

from core.cv.preprocessor import ImagePreprocessor
from core.cv.stabilizer import DiskStabilizer
from core.cv.quantizer import TrackQuantizer
from core.cv.visualizer import DetectionVisualizer
from core.audio.midi_builder import MidiBuilder
from core.audio.synthesizer import AudioSynthesizer
from core.storage.json_db import JsonDatabase


class DiskOmrPipeline:
    """
    Complete end-to-end processing pipeline:
    Image -> Contour/Center detection -> Outer-Rim Stabilization -> Track Binning -> MIDI & Audio
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.upload_dir = os.path.join(base_dir, "data", "uploads")
        self.output_dir = os.path.join(base_dir, "data", "outputs")
        self.soundfont_dir = os.path.join(base_dir, "data", "soundfonts")
        self.db_path = os.path.join(base_dir, "data", "db.json")

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.soundfont_dir, exist_ok=True)

        self.db = JsonDatabase(self.db_path)

    def process_disk_image(
        self,
        image_path: str,
        title: str = "Antique Disk Track",
        disk_radius_cm: float = 19.0,
        tempo_bpm: float = 5.3,
        note_duration: float = 0.5,
        instrument_program: int = 11,
        reverse_direction: bool = True,
        soundfont_name: Optional[str] = None,
        start_angle: float = 0.0
    ) -> Dict[str, Any]:
        """Runs full OMR and synthesis pipeline and registers result in database."""
        item_id = str(uuid.uuid4())[:8]

        # 1. Load image and extract contours & center
        gray_img = ImagePreprocessor.load_grayscale(image_path)
        prep = ImagePreprocessor.process_contours(gray_img, disk_radius_cm=disk_radius_cm)

        cropped_img = prep["cropped_img"]
        center = prep["center"]
        contours = prep["contours"]
        center_perimeter = prep["center_perimeter"]
        pred_center = prep["pred_center"]
        outer_radius_px = prep["outer_radius_px"]

        # 2. Extract point coordinates and deduplicate
        all_points = []
        for pt in prep["points_list"]:
            cx, cy, idx = pt[0], pt[1], pt[2]
            dist = math_dist((cx, cy), pred_center)
            all_points.append([cx, cy, idx, dist])

        dedup_points = DiskStabilizer.deduplicate_points(all_points, threshold_distance=10.0)

        # 3. Classify points into melody holes and outer reference teeth
        outer_thresh_dist = outer_radius_px * (15.7 / disk_radius_cm)
        melody_dots, outer_dots, split_dots = DiskStabilizer.classify_points_by_region(
            dedup_points=dedup_points,
            contours=contours,
            center_perimeter=center_perimeter,
            outer_threshold_dist=outer_thresh_dist,
            center=center
        )
        outer_dots = DiskStabilizer.deduplicate_points(outer_dots, threshold_distance=10.0)

        boundary_contour = prep.get("boundary_contour")

        # 4. Harmonic & Elliptical Geometric stabilization against disc wobble/eccentricity
        upd_top, upd_main, mean_line, diagnostics = DiskStabilizer.stabilize_with_harmonic_model(
            melody_dots=melody_dots,
            outer_dots=outer_dots,
            disc_boundary_contour=boundary_contour,
            center=center
        )
        upd_main = rotate_polar_angles(upd_main, start_angle)
        upd_top = rotate_polar_angles(upd_top, start_angle)

        # 5. Quantize stabilized holes into 78 note tracks using Adaptive Density Peak Alignment
        punched_card = TrackQuantizer.bin_holes_to_tracks(
            stabilized_dots=upd_main,
            mean_line_radius=mean_line,
            disk_radius_cm=disk_radius_cm,
            row_count=78,
            use_density_calibration=True
        )
        cleaned_card = TrackQuantizer.filter_close_punches(punched_card)

        # Total detected note count
        total_notes = sum(len(row) for row in cleaned_card)

        # 6. Save clean cropped image (for interactive Canvas editor) and visual debug preview
        clean_filename = f"{item_id}_clean.png"
        clean_path = os.path.join(self.output_dir, clean_filename)
        import cv2
        cv2.imwrite(clean_path, cropped_img)

        preview_filename = f"{item_id}_preview.png"
        preview_path = os.path.join(self.output_dir, preview_filename)
        DetectionVisualizer.draw_detections(
            cropped_img=cropped_img,
            center=center,
            melody_dots=melody_dots,
            outer_dots=outer_dots,
            split_dots=split_dots,
            output_path=preview_path
        )

        # 7. Generate MIDI File
        midi_filename = f"{item_id}.mid"
        midi_path = os.path.join(self.output_dir, midi_filename)
        MidiBuilder.create_midi(
            punched_card=cleaned_card,
            output_filepath=midi_path,
            tempo_bpm=tempo_bpm,
            note_duration=note_duration,
            instrument_program=instrument_program,
            reverse_direction=reverse_direction,
            volume_mode="radial",
            inner_volume=35,
            outer_volume=100
        )

        # 8. Synthesize Audio (WAV & MP3)
        wav_filename = f"{item_id}.wav"
        wav_path = os.path.join(self.output_dir, wav_filename)
        sf_path = os.path.join(self.soundfont_dir, soundfont_name) if soundfont_name else None

        AudioSynthesizer.synthesize_wav(
            midi_path=midi_path,
            wav_path=wav_path,
            soundfont_path=sf_path,
            punched_card=cleaned_card,
            pitch_map=TrackQuantizer.DEFAULT_PITCH_MAP,
            reverse_direction=reverse_direction,
            tempo_bpm=tempo_bpm,
            volume_mode="radial",
            inner_volume=35,
            outer_volume=100
        )

        mp3_filename = f"{item_id}.mp3"
        mp3_path = os.path.join(self.output_dir, mp3_filename)
        AudioSynthesizer.convert_to_mp3(wav_path, mp3_path)

        # 9. Store record in JSON database
        record = {
            "id": item_id,
            "title": title,
            "original_filename": os.path.basename(image_path),
            "parameters": {
                "disk_radius_cm": disk_radius_cm,
                "tempo_bpm": tempo_bpm,
                "note_duration": note_duration,
                "instrument_program": instrument_program,
                "reverse_direction": reverse_direction,
                "start_angle": float(start_angle),
                "soundfont_name": soundfont_name
            },
            "stats": {
                "melody_holes_detected": len(melody_dots),
                "outer_teeth_detected": len(outer_dots),
                "total_playable_notes": total_notes,
                "active_tracks_count": sum(1 for row in cleaned_card if len(row) > 0)
            },
            "points": {
                "center": [int(center[0]), int(center[1])],
                "start_angle": float(start_angle),
                "melody_dots": [[float(d[0]), float(d[1])] for d in melody_dots],
                "outer_dots": [[float(d[0]), float(d[1])] for d in outer_dots],
                "image_width": int(cropped_img.shape[1]),
                "image_height": int(cropped_img.shape[0])
            },
            "files": {
                "image": f"/static/outputs/{clean_filename}",
                "original_upload": f"/static/uploads/{os.path.basename(image_path)}",
                "preview": f"/static/outputs/{preview_filename}",
                "midi": f"/static/outputs/{midi_filename}",
                "wav": f"/static/outputs/{wav_filename}",
                "mp3": f"/static/outputs/{mp3_filename}" if os.path.isfile(mp3_path) else None
            }
        }

        self.db.insert(record)
        return record

    def recompute_interactive(
        self,
        disk_id: str,
        center: List[float],
        melody_dots: List[List[float]],
        outer_dots: List[List[float]],
        disk_radius_cm: float = 19.0,
        tempo_bpm: float = 5.3,
        note_duration: float = 0.5,
        instrument_program: int = 11,
        reverse_direction: bool = True,
        start_angle: float = 0.0
    ) -> Dict[str, Any]:
        """
        Recomputes calibration, tracks, MIDI, and Audio from user-edited canvas coordinates.
        """
        record = self.db.get_by_id(disk_id)
        if not record:
            raise FileNotFoundError(f"Disk with ID {disk_id} not found in database.")

        melody_pts = [[d[0], d[1], idx] for idx, d in enumerate(melody_dots)]
        outer_pts = [[d[0], d[1], idx] for idx, d in enumerate(outer_dots)]

        # Run Harmonic stabilization on user-defined points
        upd_top, upd_main, mean_line, diagnostics = DiskStabilizer.stabilize_with_harmonic_model(
            melody_dots=melody_pts,
            outer_dots=outer_pts,
            disc_boundary_contour=None,
            center=(center[0], center[1])
        )
        upd_main = rotate_polar_angles(upd_main, start_angle)
        upd_top = rotate_polar_angles(upd_top, start_angle)

        # Quantize into 78 note tracks
        punched_card = TrackQuantizer.bin_holes_to_tracks(
            stabilized_dots=upd_main,
            mean_line_radius=mean_line,
            disk_radius_cm=disk_radius_cm,
            row_count=78,
            use_density_calibration=True
        )
        cleaned_card = TrackQuantizer.filter_close_punches(punched_card)
        total_notes = sum(len(row) for row in cleaned_card)

        # Re-generate MIDI File
        midi_filename = f"{disk_id}.mid"
        midi_path = os.path.join(self.output_dir, midi_filename)
        MidiBuilder.create_midi(
            punched_card=cleaned_card,
            output_filepath=midi_path,
            tempo_bpm=tempo_bpm,
            note_duration=note_duration,
            instrument_program=instrument_program,
            reverse_direction=reverse_direction,
            volume_mode="radial",
            inner_volume=35,
            outer_volume=100
        )

        # Re-synthesize Audio
        wav_filename = f"{disk_id}.wav"
        wav_path = os.path.join(self.output_dir, wav_filename)
        AudioSynthesizer.synthesize_wav(
            midi_path=midi_path,
            wav_path=wav_path,
            punched_card=cleaned_card,
            pitch_map=TrackQuantizer.DEFAULT_PITCH_MAP,
            reverse_direction=reverse_direction,
            tempo_bpm=tempo_bpm,
            volume_mode="radial",
            inner_volume=35,
            outer_volume=100
        )

        # Update record in JSON DB
        record["parameters"].update({
            "disk_radius_cm": disk_radius_cm,
            "tempo_bpm": tempo_bpm,
            "note_duration": note_duration,
            "instrument_program": instrument_program,
            "reverse_direction": reverse_direction,
            "start_angle": float(start_angle)
        })
        record["stats"] = {
            "melody_holes_detected": len(melody_dots),
            "outer_teeth_detected": len(outer_dots),
            "total_playable_notes": total_notes,
            "active_tracks_count": sum(1 for row in cleaned_card if len(row) > 0)
        }
        record["points"] = {
            "center": [int(center[0]), int(center[1])],
            "start_angle": float(start_angle),
            "melody_dots": [[float(d[0]), float(d[1])] for d in melody_dots],
            "outer_dots": [[float(d[0]), float(d[1])] for d in outer_dots],
            "image_width": record.get("points", {}).get("image_width", 1000),
            "image_height": record.get("points", {}).get("image_height", 1000)
        }

        # Update and save to db
        self.db.delete(disk_id)
        self.db.insert(record)
        return record

    def save_interactive_settings(
        self,
        disk_id: str,
        title: str,
        center: List[float],
        melody_dots: List[List[float]],
        outer_dots: List[List[float]],
        start_angle: float,
        disk_radius_cm: float,
        tempo_bpm: float,
        note_duration: float,
        instrument_program: int,
        reverse_direction: bool
    ) -> Dict[str, Any]:
        """Persists Human-in-the-Loop edits without regenerating audio files."""
        record = self.db.get_by_id(disk_id)
        if not record:
            raise FileNotFoundError(f"Disk with ID {disk_id} not found in database.")

        changes = {
            "title": title,
            "parameters": {
                **record.get("parameters", {}),
                "disk_radius_cm": disk_radius_cm,
                "tempo_bpm": tempo_bpm,
                "note_duration": note_duration,
                "instrument_program": instrument_program,
                "reverse_direction": reverse_direction,
                "start_angle": float(start_angle)
            },
            "points": {
                **record.get("points", {}),
                "center": [int(center[0]), int(center[1])],
                "start_angle": float(start_angle),
                "melody_dots": [[float(d[0]), float(d[1])] for d in melody_dots],
                "outer_dots": [[float(d[0]), float(d[1])] for d in outer_dots]
            }
        }
        updated = self.db.update(disk_id, changes)
        if not updated:
            raise FileNotFoundError(f"Disk with ID {disk_id} not found in database.")
        return updated


def math_dist(p1, p2):
    import math
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def rotate_polar_angles(points, start_angle):
    """Moves the user-selected disk start line to angle zero."""
    if points is None or len(points) == 0:
        return points
    rotated = points.copy()
    rotated[:, 0] = (rotated[:, 0] - float(start_angle)) % (2.0 * np.pi)
    return rotated
