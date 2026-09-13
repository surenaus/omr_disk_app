import json

from core.cv.quantizer import TrackQuantizer
from core.storage.json_db import JsonDatabase


def test_pitch_map_has_78_tracks():
    assert len(TrackQuantizer.DEFAULT_PITCH_MAP) == 78
    assert TrackQuantizer.DEFAULT_PITCH_MAP[25] == 0
    assert TrackQuantizer.DEFAULT_PITCH_MAP[51] == 0


def test_json_database_insert_update(tmp_path):
    database = JsonDatabase(str(tmp_path / "db.json"))
    created = database.insert({"title": "Test disk"})

    assert created["title"] == "Test disk"
    assert created["created_at"] == created["updated_at"]

    updated = database.update(created["id"], {"title": "Renamed disk"})
    assert updated["title"] == "Renamed disk"
    assert updated["created_at"] == created["created_at"]
    assert "updated_at" in updated

    stored = json.loads((tmp_path / "db.json").read_text(encoding="utf-8"))
    assert stored["disks"][0]["title"] == "Renamed disk"
