import os
import math
import struct
import wave
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class AudioSynthesizer:
    """
    Synthesizes audio (.wav / .mp3) from MIDI files using SoundFonts (FluidSynth)
    with a built-in pure-Python acoustic music-box chime synthesis fallback.
    """

    @classmethod
    def synthesize_wav(
        cls,
        midi_path: str,
        wav_path: str,
        soundfont_path: Optional[str] = None,
        sample_rate: int = 44100,
        punched_card: Optional[List[List[float]]] = None,
        pitch_map: Optional[List[int]] = None,
        reverse_direction: bool = True,
        tempo_bpm: float = 5.3,
        disk_revolution_seconds: Optional[float] = None,
        volume_mode: str = "radial",
        inner_volume: int = 35,
        outer_volume: int = 100
    ) -> str:
        """Renders MIDI to WAV using FluidSynth or fallback tone generator."""
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        # 1. Try FluidSynth if soundfont exists
        if soundfont_path and os.path.isfile(soundfont_path):
            try:
                from midi2audio import FluidSynth
                fs = FluidSynth(soundfont_path, sample_rate=sample_rate)
                fs.midi_to_audio(midi_path, wav_path)
                if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
                    logger.info("Successfully synthesized audio with FluidSynth.")
                    return wav_path
            except Exception as e:
                logger.warning(f"FluidSynth rendering failed, using acoustic fallback synth: {e}")

        # 2. Pure Python Music Box Sound Generator Fallback
        cls._render_midi_fallback(
            midi_path, 
            wav_path, 
            sample_rate=sample_rate,
            punched_card=punched_card,
            pitch_map=pitch_map,
            reverse_direction=reverse_direction,
            tempo_bpm=tempo_bpm,
            disk_revolution_seconds=disk_revolution_seconds,
            volume_mode=volume_mode,
            inner_volume=inner_volume,
            outer_volume=outer_volume
        )
        return wav_path

    @classmethod
    def convert_to_mp3(cls, wav_path: str, mp3_path: str) -> Optional[str]:
        """Converts WAV to MP3 using pydub if ffmpeg is available."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3")
            return mp3_path
        except Exception as e:
            logger.info(f"MP3 conversion skipped/unavailable: {e}")
            return None

    @classmethod
    def _write_empty_wav(cls, wav_path: str, sample_rate: int = 44100):
        """Generates a 1-second silent WAV file."""
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            raw_frames = struct.pack(f'<{sample_rate}h', *([0] * sample_rate))
            wav_file.writeframes(raw_frames)

    @classmethod
    def _render_midi_fallback(
        cls, 
        midi_path: str, 
        wav_path: str, 
        sample_rate: int = 44100, 
        punched_card: Optional[List[List[float]]] = None,
        pitch_map: Optional[List[int]] = None,
        reverse_direction: bool = True,
        tempo_bpm: float = 5.3,
        disk_revolution_seconds: Optional[float] = None,
        volume_mode: str = "radial",
        inner_volume: int = 35,
        outer_volume: int = 100
    ):
        """
        Synthesizes resonant metallic music-box chime notes directly to 16-bit PCM WAV.
        """
        events = []

        # If disk_revolution_seconds is provided, use it (e.g. 45s - 60s per full rotation).
        # Otherwise compute from tempo_bpm: in MIDI beat_time = angle, and seconds_per_beat = 60.0 / tempo_bpm.
        if disk_revolution_seconds is not None and disk_revolution_seconds > 0:
            time_scale = disk_revolution_seconds / (2.0 * math.pi)
        else:
            time_scale = 60.0 / max(0.1, tempo_bpm)

        if punched_card is not None and pitch_map is not None:
            # Direct note event generation from punched card data
            row_count = min(len(punched_card), len(pitch_map))
            for row_idx in range(row_count):
                pitch = pitch_map[row_idx]
                if pitch <= 0 or row_idx in [25, 51]:
                    continue
                if volume_mode == "radial" and row_count > 1:
                    note_velocity = round(inner_volume + (outer_volume - inner_volume) * row_idx / (row_count - 1))
                else:
                    note_velocity = outer_volume
                for angle in punched_card[row_idx]:
                    raw_angle = (2.0 * math.pi - angle) if reverse_direction else angle
                    event_time = max(0.0, float(raw_angle)) * time_scale
                    events.append((event_time, pitch, note_velocity))
        else:
            try:
                import mido
                mid = mido.MidiFile(midi_path)
                current_time = 0.0
                for msg in mid:
                    current_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        events.append((current_time, msg.note, msg.velocity))
            except Exception:
                events = [(i * 0.25, 60 + (i % 12), 100) for i in range(16)]

        if not events:
            cls._write_empty_wav(wav_path, sample_rate)
            return

        total_duration = max([t for t, _, _ in events]) + 2.5
        total_samples = int(total_duration * sample_rate)
        buffer = [0.0] * total_samples

        for start_sec, note_num, velocity in events:
            freq = 440.0 * (2.0 ** ((note_num - 69.0) / 12.0))
            start_sample = int(start_sec * sample_rate)
            note_len_samples = int(1.5 * sample_rate)  # 1.5 second resonant decay

            vol = (velocity / 127.0) * 0.35

            for s in range(note_len_samples):
                idx = start_sample + s
                if idx >= total_samples:
                    break
                t = s / float(sample_rate)
                # Exponential decay envelope characteristic of metal music box tines
                decay = math.exp(-3.8 * t)
                # Fundamental tone + slight metallic 2nd and 3rd harmonics
                sample_val = vol * decay * (
                    0.75 * math.sin(2.0 * math.pi * freq * t) +
                    0.20 * math.sin(2.0 * math.pi * freq * 2.0 * t) +
                    0.05 * math.sin(2.0 * math.pi * freq * 3.0 * t)
                )
                buffer[idx] += sample_val

        # Normalize and convert to 16-bit PCM
        max_val = max(max(abs(x) for x in buffer), 1e-5)
        scale = 30000.0 / max_val if max_val > 1.0 else 30000.0
        pcm_data = [int(max(-32767, min(32767, val * scale))) for val in buffer]

        # Write WAV file
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            raw_frames = struct.pack(f'<{len(pcm_data)}h', *pcm_data)
            wav_file.writeframes(raw_frames)
