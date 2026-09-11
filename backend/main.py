from __future__ import annotations

import sys
from pathlib import Path

# Add project root and backend folder to sys.path
PROJECT_ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT_DIR))
sys.path.insert(0, str(PROJECT_ROOT_DIR / "backend"))

import hashlib
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Literal

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.services.analytics_engine import (
    get_analytics,
    update_analytics,
)
from backend.services.audio_pipeline import process_audio
from backend.services.docx_processor import process_docx
from backend.services.forensic_pipeline import analyze_text
from backend.services.image_pipeline import process_image
from backend.services.pdf_processor import process_pdf
from backend.services.report_generator import generate_pdf_report

# =====================================================
# CONTINUOUS LEARNING FEEDBACK IMPORTS
# =====================================================
from continuous_learning.image.replay_manager import append_replay_record
from continuous_learning.image.user_feedback import process_feedback as process_image_feedback

from continuous_learning.text.feedback_manager import save_feedback as save_text_feedback
from continuous_learning.audio.feedback_manager import save_feedback as save_audio_feedback
from continuous_learning.pending_manager import (
    save_pending_item,
    get_all_pending_items,
    get_pending_count,
    resolve_pending_item
)
class TextFeedbackSchema(BaseModel):
    sentence: str
    predicted_label: str
    confidence: float
    verified_label: str
# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = PROJECT_ROOT / "uploads"
AUDIO_UPLOAD_DIR = UPLOAD_DIR / "audio"
BACKEND_UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
REPORT_DIR = PROJECT_ROOT / "reports"

AUDIO_VISUAL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "audio_visuals"
)

WAVEFORM_DIR = AUDIO_VISUAL_DIR / "waveforms"
SPECTROGRAM_DIR = AUDIO_VISUAL_DIR / "spectrograms"
AUDIO_HEATMAP_DIR = AUDIO_VISUAL_DIR / "heatmaps"

for directory in (
    UPLOAD_DIR,
    AUDIO_UPLOAD_DIR,
    BACKEND_UPLOAD_DIR,
    REPORT_DIR,
    WAVEFORM_DIR,
    SPECTROGRAM_DIR,
    AUDIO_HEATMAP_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# =========================================================
# SINGLE FASTAPI APP INSTANCE
# =========================================================

app = FastAPI(
    title="F.O.R.G.E. Gateway API",
    description="Multimodal explainable AI platform for text, image and audio forensic analysis.",
    version="2.0.0",
)
# =========================================================
# PYDANTIC SCHEMAS (REQUIRED FOR SWAGGER DOCS)
# =========================================================

class ImageFeedbackRequest(BaseModel):
    image_id: Optional[str] = None

    image_path: str

    predicted_label: str

    verified_label: str

    confidence: float

    user_feedback: str

class TextFeedbackRequest(BaseModel):
    sentence_id: Optional[str] = Field(default=None, example="txt_a1b2c3d4")
    sentence: str = Field(..., example="This text looks synthetic.")
    predicted_label: str = Field(default="Fake", example="Fake")
    verified_label: str = Field(default="Real", example="Real")
    confidence: float = Field(default=85.0, example=85.0)

class AudioFeedbackRequest(BaseModel):
    audio_path: str = Field(..., example="/uploads/audio/sample.wav")
    audio_name: Optional[str] = Field(default="sample.wav", example="sample.wav")
    predicted_label: str = Field(default="Fake", example="Fake")
    verified_label: str = Field(default="Real", example="Real")
    confidence: float = Field(default=90.0, example=90.0)
    model_version: Optional[str] = Field(default="v1", example="v1")
# =========================================================
# STATIC FILES MOUNTS
# =========================================================

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/backend-uploads", StaticFiles(directory=str(BACKEND_UPLOAD_DIR)), name="backend_uploads")
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")
app.mount("/audio-visuals", StaticFiles(directory=str(AUDIO_VISUAL_DIR)), name="audio_visuals")

# =========================================================
# CORS MIDDLEWARE
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# CONSTANTS & HELPER MODELS
# =========================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
TEXT_EXTENSIONS = {".txt", ".docx", ".pdf"}

class VerifyLaterRequest(BaseModel):
    modality: Literal["image", "audio", "text"]
    identifier: str
    prediction: str
    confidence: float
    filepath: Optional[str] = None

class PendingVerifyRequest(BaseModel):
    item_id: str
    final_label: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def generate_case_id(modality: str) -> str:
    prefix = {"text": "TXT", "image": "IMG", "audio": "AUD"}.get(modality, "GEN")
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    random_part = uuid.uuid4().hex[:8].upper()
    return f"FORGE-{prefix}-{date_part}-{random_part}"

def sanitize_filename(filename: str) -> str:
    safe_name = os.path.basename(filename).replace(" ", "_")
    return f"{uuid.uuid4().hex}_{safe_name}"

def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def save_uploaded_file(uploaded_file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

def build_evidence_metadata(*, file_path: Optional[Path], original_filename: Optional[str], modality: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "original_filename": original_filename or "direct_text_input",
        "modality": modality,
        "mime_type": mime_type or "text/plain",
        "analysis_timestamp_utc": utc_timestamp(),
        "size_bytes": None,
        "sha256": None,
    }
    if file_path and file_path.exists():
        evidence["size_bytes"] = file_path.stat().st_size
        evidence["sha256"] = calculate_sha256(file_path)
    return evidence

def normalize_probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0 <= number <= 1:
        number *= 100
    return round(max(0.0, min(100.0, number)), 2)

def attach_standard_contract(*, result: Dict[str, Any], modality: str, case_id: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    fake_prob = normalize_probability(result.get("raw_ai_probability") or result.get("raw_probability_fake") or result.get("risk_score") or 0)
    real_prob = result.get("raw_human_probability") or result.get("raw_probability_real")
    if real_prob is None:
        real_prob = 100 - fake_prob
    real_prob = normalize_probability(real_prob)

    result["case_id"] = case_id
    result["evidence"] = evidence
    result["modality"] = modality
    result["file_type"] = modality
    result["probabilities"] = {"ai": fake_prob, "human": real_prob}
    result["analysis_version"] = "FORGE-XAI-2.0"

    confidence = normalize_probability(result.get("confidence", 0))
    if confidence >= 90:
        decision_strength = "VERY STRONG"
    elif confidence >= 75:
        decision_strength = "STRONG"
    elif confidence >= 60:
        decision_strength = "MODERATE"
    else:
        decision_strength = "LIMITED"

    result["decision_strength"] = decision_strength
    return result

def create_report(result: Dict[str, Any]) -> None:
    case_id = result.get("case_id", generate_case_id(result.get("modality", "general")))
    report_name = f"{case_id}_{datetime.now(timezone.utc).strftime('%H%M%S')}.pdf"
    report_path = REPORT_DIR / report_name
    try:
        generate_pdf_report(result, str(report_path))
        result["pdf_report"] = f"/reports/{report_name}"
    except Exception as error:
        result["pdf_report_error"] = str(error)

def update_module_analytics(modality: str, result: Dict[str, Any]) -> None:
    if result.get("error"):
        return
    update_analytics(modality, result.get("prediction", "UNKNOWN"), result.get("confidence", 0))

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def home():
    return {"message": "FORGE Backend Running", "version": "2.0.0", "status": "online"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "FORGE API", "timestamp": utc_timestamp()}

# =========================================================
# MAIN ANALYZE ENDPOINT
# =========================================================

@app.post("/analyze")
async def analyze(file: Optional[UploadFile] = File(default=None), text: str = Form(default="")):
    try:
        result: Optional[Dict[str, Any]] = None
        modality = "text"
        file_path: Optional[Path] = None
        original_filename: Optional[str] = None
        mime_type: Optional[str] = None
        uploaded_public_url: Optional[str] = None

        if text.strip():
            modality = "text"
            result = analyze_text(text.strip())
        elif file is not None:
            if not file.filename:
                return {"error": "Empty filename"}
            original_filename = file.filename
            mime_type = file.content_type
            extension = Path(original_filename).suffix.lower()
            stored_filename = sanitize_filename(original_filename)

            if extension in IMAGE_EXTENSIONS:
                modality = "image"
                file_path = UPLOAD_DIR / stored_filename
                save_uploaded_file(file, file_path)
                result = process_image(str(file_path))
                uploaded_public_url = f"/uploads/{stored_filename}"

                # --- CONTINUOUS LEARNING FIX ---
                # Save public URL and log features into replay_dataset.csv
                result["uploaded_file"] = uploaded_public_url
                image_id = append_replay_record(result)
                result["image_id"] = image_id
                # -------------------------------"

            elif extension in AUDIO_EXTENSIONS:
                modality = "audio"
                file_path = AUDIO_UPLOAD_DIR / stored_filename
                save_uploaded_file(file, file_path)
                result = process_audio(str(file_path))
                uploaded_public_url = f"/uploads/audio/{stored_filename}"
                if isinstance(result, dict):
                    result["uploaded_file"] = uploaded_public_url

            elif extension == ".docx":
                modality = "text"
                file_path = UPLOAD_DIR / stored_filename
                save_uploaded_file(file, file_path)
                result = process_docx(str(file_path))
                uploaded_public_url = f"/uploads/{stored_filename}"

            elif extension == ".pdf":
                modality = "text"
                file_path = UPLOAD_DIR / stored_filename
                save_uploaded_file(file, file_path)
                result = process_pdf(str(file_path))
                uploaded_public_url = f"/uploads/{stored_filename}"

            elif extension == ".txt":
                modality = "text"
                file_path = UPLOAD_DIR / stored_filename
                save_uploaded_file(file, file_path)
                text_content = file_path.read_text(encoding="utf-8", errors="ignore")
                result = analyze_text(text_content)
                uploaded_public_url = f"/uploads/{stored_filename}"
            else:
                return {"error": f"Unsupported file type: {extension}"}
        else:
            return {"error": "No input provided"}

        if result is None or result.get("error"):
            return result or {"error": "Analysis failed"}

        case_id = generate_case_id(modality)
        evidence = build_evidence_metadata(file_path=file_path, original_filename=original_filename, modality=modality, mime_type=mime_type)
        result = attach_standard_contract(result=result, modality=modality, case_id=case_id, evidence=evidence)

        if uploaded_public_url:
            result["uploaded_file"] = uploaded_public_url

        update_module_analytics(modality, result)
        create_report(result)
        #from continuous_learning.image.replay_manager import append_replay_record

        #if modality == "image":
            
            #result["image_id"] = image_id
        return result

    except Exception as error:
        return {"error": str(error)}
# =========================================================
# CONTINUOUS LEARNING FEEDBACK ENDPOINTS
# =========================================================
@app.post("/feedback/image")
@app.post("/image-feedback")
async def handle_image_feedback(req: ImageFeedbackRequest):
    data = req.dict()
    image_id = data.get("image_id")
    verified_label = data.get("verified_label", "HUMAN")
    user_feedback_text = data.get("user_feedback", "Agree")

    if not image_id:
        raise HTTPException(status_code=400, detail="Missing image_id in feedback request")

    # Update replay_dataset.csv using user_feedback.py
    result = process_image_feedback(
        image_id=image_id, 
        verified_label=verified_label, 
        user_feedback=user_feedback_text
    )

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))

    return {
        "status": "success",
        "result": result
    }
@app.post("/feedback/text")
@app.post("/text-feedback")
async def handle_text_feedback(data: dict):
    # Retrieve payload fields
    sentence = data.get("sentence", "")
    predicted_label = data.get("predicted_label", "Fake")
    confidence = data.get("confidence", 0.0)
    verified_label = data.get("verified_label", "HUMAN")
    response = save_text_feedback(
            sentence=sentence,
            predicted_label=predicted_label,
            confidence=confidence,
            verified_label=verified_label
        )
        
    return response
    
@app.post("/feedback/audio")
@app.post("/audio-feedback")
async def handle_audio_feedback(req: AudioFeedbackRequest):  # <--- MUST USE req: AudioFeedbackRequest
    try:
        result = save_audio_feedback(
            audio_path=req.audio_path,
            audio_name=req.audio_name or Path(req.audio_path).name,
            predicted_label=req.predicted_label,
            verified_label=req.verified_label,
            confidence=req.confidence,
            model_version=req.model_version or "v1",
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/verify-later")
async def verify_later(req: VerifyLaterRequest):
    item_id = save_pending_item(
        modality=req.modality,
        identifier=req.identifier,
        prediction=req.prediction,
        confidence=req.confidence,
        source_filepath=req.filepath
    )
    return {"status": "success", "message": "Sample moved to pending verification", "item_id": item_id}

@app.get("/pending/count")
async def pending_count():
    return {"count": get_pending_count()}

@app.get("/pending/list")
async def pending_list():
    return {"items": get_all_pending_items()}

@app.post("/pending/verify")
async def pending_verify(req: PendingVerifyRequest):
    record = resolve_pending_item(req.item_id, req.final_label)
    if not record:
        raise HTTPException(status_code=404, detail="Pending item not found")

    modality = record["modality"]
    identifier = record["identifier"]

    if modality == "image":
        process_feedback(image_id=identifier, verified_label=req.final_label)
    elif modality == "audio":
        save_audio_feedback(audio_path=identifier, audio_name=Path(identifier).name, predicted_label="Fake", verified_label=req.final_label, confidence=0.8)
    elif modality == "text":
        save_text_feedback(sentence=identifier, predicted_label="Fake", confidence=0.8, verified_label=req.final_label)

    return {"status": "success", "message": f"Verified as {req.final_label} and sent to {modality} retraining pipeline."}

@app.get("/analytics")
def analytics():
    return get_analytics()