from __future__ import annotations
import hashlib, mimetypes, os, uuid
from datetime import datetime, timezone

def generate_case_id(modality: str) -> str:
    prefix = {"text":"TXT","image":"IMG","audio":"AUD","video":"VID"}.get(modality.lower(),"GEN")
    return f"FORGE-{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

def sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def build_evidence_metadata(file_path, original_filename, modality):
    data = {
        "original_filename": original_filename or "direct_text_input",
        "modality": modality,
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if file_path and os.path.exists(file_path):
        data.update({
            "stored_path": file_path,
            "size_bytes": os.path.getsize(file_path),
            "mime_type": mimetypes.guess_type(original_filename or file_path)[0] or "application/octet-stream",
            "sha256": sha256_file(file_path),
        })
    else:
        data.update({"stored_path": None, "size_bytes": None, "mime_type": "text/plain", "sha256": None})
    return data
