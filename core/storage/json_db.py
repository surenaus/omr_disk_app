import os
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional


class JsonDatabase:
    """
    Lightweight, atomic JSON database to store processed disk metadata,
    calibrations, and artifact paths.
    """

    def __init__(self, db_filepath: str):
        self.db_filepath = db_filepath
        self._ensure_db()

    def _ensure_db(self):
        os.makedirs(os.path.dirname(self.db_filepath), exist_ok=True)
        if not os.path.isfile(self.db_filepath):
            with open(self.db_filepath, "w", encoding="utf-8") as f:
                json.dump({"disks": []}, f, indent=2)

    def _read_data(self) -> Dict[str, Any]:
        try:
            with open(self.db_filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"disks": []}

    def _write_data(self, data: Dict[str, Any]):
        temp_file = self.db_filepath + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, self.db_filepath)

    def insert(self, record: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "id" not in record:
            record["id"] = str(uuid.uuid4())[:8]
        if "created_at" not in record:
            record["created_at"] = now
        record["updated_at"] = now
        
        data = self._read_data()
        data["disks"].insert(0, record)  # most recent first
        self._write_data(data)
        return record

    def list_all(self) -> List[Dict[str, Any]]:
        data = self._read_data()
        return data.get("disks", [])

    def get_by_id(self, disk_id: str) -> Optional[Dict[str, Any]]:
        disks = self.list_all()
        for d in disks:
            if d.get("id") == disk_id:
                return d
        return None

    def update(self, disk_id: str, changes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates one record while preserving created_at and refreshing updated_at."""
        data = self._read_data()
        for record in data.get("disks", []):
            if record.get("id") == disk_id:
                created_at = record.get("created_at")
                record.update(changes)
                record["id"] = disk_id
                if created_at:
                    record["created_at"] = created_at
                record["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._write_data(data)
                return record
        return None

    def delete(self, disk_id: str) -> bool:
        data = self._read_data()
        initial_len = len(data.get("disks", []))
        data["disks"] = [d for d in data.get("disks", []) if d.get("id") != disk_id]
        if len(data["disks"]) < initial_len:
            self._write_data(data)
            return True
        return False
