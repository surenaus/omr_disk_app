import math
import os
from typing import List, Optional
from midiutil import MIDIFile


class MidiBuilder:
    """Generates standard MIDI Format 1 files from punched disk note tracks."""

    @staticmethod
    def create_midi(
        punched_card: List[List[float]],
        output_filepath: str,
        pitch_map: Optional[List[int]] = None,
        tempo_bpm: float = 5.3,
        note_duration: float = 0.5,
        instrument_program: int = 11,  # 11 = Music Box in General MIDI
        reverse_direction: bool = True,
        volume: int = 100,
        volume_mode: str = "radial",
        inner_volume: int = 35,
        outer_volume: int = 100,
        exclude_tracks: Optional[List[int]] = None
    ) -> str:
        """
        Creates a .mid file mapping angular positions to time beats.
        """
        if exclude_tracks is None:
            exclude_tracks = [25, 51]  # Standard spacer tracks

        if pitch_map is None:
            from core.cv.quantizer import TrackQuantizer
            pitch_map = TrackQuantizer.DEFAULT_PITCH_MAP

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        midi = MIDIFile(1)
        track = 0
        channel = 0
        time_offset = 0.0

        midi.addTempo(track, time_offset, max(1.0, tempo_bpm))
        midi.addProgramChange(track, channel, time_offset, instrument_program)

        row_count = min(len(punched_card), len(pitch_map))
        
        for row_idx in range(row_count):
            if row_idx in exclude_tracks:
                continue

            midi_pitch = pitch_map[row_idx]
            if midi_pitch <= 0:
                continue

            angles = punched_card[row_idx]
            if volume_mode == "radial" and row_count > 1:
                row_volume = round(inner_volume + (outer_volume - inner_volume) * row_idx / (row_count - 1))
            else:
                row_volume = volume

            for angle in angles:
                if reverse_direction:
                    # Clockwise disk rotation maps angle 2*pi - theta to playback timeline
                    note_time = (2.0 * math.pi - angle)
                else:
                    note_time = angle

                # Scale angle to beats
                beat_time = max(0.0, float(note_time))
                midi.addNote(track, channel, midi_pitch, beat_time, note_duration, row_volume)

        with open(output_filepath, "wb") as f:
            midi.writeFile(f)

        return output_filepath
