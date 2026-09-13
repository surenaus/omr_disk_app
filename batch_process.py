import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.pipeline import DiskOmrPipeline

def run_batch():
    pipeline = DiskOmrPipeline(BASE_DIR)
    upload_dir = pipeline.upload_dir
    images = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images in uploads directory:\n")

    for img_name in sorted(images):
        img_path = os.path.join(upload_dir, img_name)
        title = os.path.splitext(img_name)[0]
        print(f"========================================")
        print(f"Processing: {img_name}")
        print(f"========================================")
        try:
            res = pipeline.process_disk_image(img_path, title=title, tempo_bpm=5.3)
            stats = res.get("stats", {})
            print(f"  [+] Status: SUCCESS")
            print(f"  [+] ID: {res['id']}")
            print(f"  [+] Melody holes detected: {stats.get('melody_holes_detected')}")
            print(f"  [+] Active note tracks: {stats.get('active_tracks_count')}/78")
            print(f"  [+] Total notes generated: {stats.get('total_playable_notes')}")
            print(f"  [+] MIDI: {res['files']['midi']}")
            print(f"  [+] WAV:  {res['files']['wav']}")
            print(f"  [+] Preview: {res['files']['preview']}\n")
        except Exception as e:
            print(f"  [-] Status: ERROR: {e}\n")

if __name__ == '__main__':
    run_batch()
