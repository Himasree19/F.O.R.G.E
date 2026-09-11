import fitz

from backend.services.forensic_pipeline import (
    analyze_text
)

# ==========================================
# PDF PROCESSOR
# ==========================================

def process_pdf(file_path):

    doc = fitz.open(file_path)

    full_text = ""

    for page in doc:

        full_text += page.get_text()

    return analyze_text(full_text)