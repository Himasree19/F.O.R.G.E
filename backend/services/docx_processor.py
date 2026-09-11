from docx import Document

from backend.services.forensic_pipeline import (
    analyze_text
)

from backend.services.docx_highlighter import (
    generate_highlighted_docx
)

from backend.services.report_generator import (
    generate_pdf_report
)

from backend.services.chart_service import (
    generate_feature_chart
)

import os

# ==========================================
# DOCX PROCESSOR
# ==========================================

def process_docx(file_path):

    doc = Document(file_path)

    full_text = []

    for para in doc.paragraphs:

        text = para.text.strip()

        if text:

            full_text.append(text)

    combined_text = "\n".join(
        full_text
    )

    # ======================================
    # ANALYZE
    # ======================================

    result = analyze_text(
        combined_text
    )

    # ======================================
    # HIGHLIGHTED DOCX
    # ======================================

    highlighted_path = os.path.join(

        "uploads",

        "highlighted_output.docx"
    )

    generate_highlighted_docx(

        combined_text,

        result["full_document"],

        highlighted_path
    )

    # ======================================
    # PDF REPORT
    # ======================================

    report_path = os.path.join(

        "reports",

        "forensic_report.pdf"
    )

    generate_pdf_report(

        result,

        report_path
    )

    # ======================================
    # FEATURE GRAPH
    # ======================================

    chart_path = os.path.join(

        "charts",

        "feature_chart.png"
    )

    generate_feature_chart(

        result[
            "parameter_contribution"
        ],

        chart_path
    )

    # ======================================
    # SAVE PATHS
    # ======================================

    result["highlighted_docx"] = (
        highlighted_path
    )

    result["pdf_report"] = (
        report_path
    )

    result["chart"] = (
        chart_path
    )

    return result