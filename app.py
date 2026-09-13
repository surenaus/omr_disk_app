import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Add current directory to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.pipeline import DiskOmrPipeline

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB upload limit

# Initialize pipeline and directories
pipeline = DiskOmrPipeline(base_dir=BASE_DIR)


@app.route('/')
def index():
    disks = pipeline.db.list_all()
    return render_template('index.html', disks=disks)


@app.route('/api/process', methods=['POST'])
def process_image():
    if 'disk_image' not in request.files:
        return jsonify({"error": "No disk image provided in request"}), 400

    file = request.files['disk_image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(pipeline.upload_dir, filename)
    file.save(upload_path)

    title = request.form.get('title', 'Antique Disc')
    disk_radius_cm = float(request.form.get('disk_radius_cm', 19.0))
    tempo_bpm = float(request.form.get('tempo_bpm', 5.3))
    note_duration = float(request.form.get('note_duration', 0.5))
    instrument_program = int(request.form.get('instrument_program', 11))
    reverse_direction = request.form.get('reverse_direction') in ['true', 'True', '1', 'on']
    start_angle = float(request.form.get('start_angle', 0.0))

    try:
        record = pipeline.process_disk_image(
            image_path=upload_path,
            title=title,
            disk_radius_cm=disk_radius_cm,
            tempo_bpm=tempo_bpm,
            note_duration=note_duration,
            instrument_program=instrument_program,
            reverse_direction=reverse_direction,
            start_angle=start_angle
        )
        return jsonify(record)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recompute', methods=['POST'])
def recompute():
    data = request.get_json()
    if not data or 'disk_id' not in data:
        return jsonify({"error": "Missing disk_id or data payload"}), 400

    disk_id = data['disk_id']
    center = data.get('center', [500, 500])
    melody_dots = data.get('melody_dots', [])
    outer_dots = data.get('outer_dots', [])
    disk_radius_cm = float(data.get('disk_radius_cm', 19.0))
    tempo_bpm = float(data.get('tempo_bpm', 5.3))
    note_duration = float(data.get('note_duration', 0.5))
    instrument_program = int(data.get('instrument_program', 11))
    reverse_direction = bool(data.get('reverse_direction', True))
    start_angle = float(data.get('start_angle', 0.0))

    try:
        updated_record = pipeline.recompute_interactive(
            disk_id=disk_id,
            center=center,
            melody_dots=melody_dots,
            outer_dots=outer_dots,
            disk_radius_cm=disk_radius_cm,
            tempo_bpm=tempo_bpm,
            note_duration=note_duration,
            instrument_program=instrument_program,
            reverse_direction=reverse_direction,
            start_angle=start_angle
        )
        return jsonify(updated_record)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/regenerate', methods=['POST'])
def regenerate():
    data = request.get_json()
    if not data or 'disk_id' not in data:
        return jsonify({"error": "Missing disk_id"}), 400

    record = pipeline.db.get_by_id(data['disk_id'])
    if not record or 'points' not in record:
        return jsonify({"error": "Saved point data not found for this disk"}), 404

    points = record['points']
    try:
        updated_record = pipeline.recompute_interactive(
            disk_id=data['disk_id'],
            center=points.get('center', [500, 500]),
            melody_dots=points.get('melody_dots', []),
            outer_dots=points.get('outer_dots', []),
            disk_radius_cm=float(data.get('disk_radius_cm', record.get('parameters', {}).get('disk_radius_cm', 19.0))),
            tempo_bpm=float(data.get('tempo_bpm', record.get('parameters', {}).get('tempo_bpm', 5.3))),
            note_duration=float(data.get('note_duration', record.get('parameters', {}).get('note_duration', 0.5))),
            instrument_program=int(data.get('instrument_program', 11)),
            reverse_direction=bool(data.get('reverse_direction', record.get('parameters', {}).get('reverse_direction', True))),
            start_angle=float(data.get('start_angle', points.get('start_angle', 0.0)))
        )
        return jsonify(updated_record)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/save-edits', methods=['POST'])
def save_edits():
    data = request.get_json()
    if not data or 'disk_id' not in data:
        return jsonify({"error": "Missing disk_id"}), 400

    try:
        record = pipeline.save_interactive_settings(
            disk_id=data['disk_id'],
            title=str(data.get('title', 'Antique Disc')).strip() or 'Antique Disc',
            center=data.get('center', [500, 500]),
            melody_dots=data.get('melody_dots', []),
            outer_dots=data.get('outer_dots', []),
            start_angle=float(data.get('start_angle', 0.0)),
            disk_radius_cm=float(data.get('disk_radius_cm', 19.0)),
            tempo_bpm=float(data.get('tempo_bpm', 5.3)),
            note_duration=float(data.get('note_duration', 0.5)),
            instrument_program=int(data.get('instrument_program', 11)),
            reverse_direction=bool(data.get('reverse_direction', True))
        )
        return jsonify(record)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/disk/<disk_id>', methods=['GET'])
def get_disk(disk_id):
    record = pipeline.db.get_by_id(disk_id)
    if record:
        return jsonify(record)
    return jsonify({"error": "Not found"}), 404


@app.route('/api/disk/<disk_id>', methods=['DELETE'])
def delete_disk(disk_id):
    success = pipeline.db.delete(disk_id)
    return jsonify({"success": success})


@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(pipeline.upload_dir, filename)


@app.route('/static/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(pipeline.output_dir, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🎵 Punched Disk OMR & Synthesizer Server running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
