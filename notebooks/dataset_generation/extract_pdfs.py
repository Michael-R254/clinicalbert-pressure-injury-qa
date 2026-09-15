"""
PDF to plain text extraction for pressure_ulcer_qa source documents.
Reads all PDFs from sources/PDFs/, extracts text, saves to sources/raw/.
Also updates sources/metadata.json with the actual documents found.
"""

import json
import re
from pathlib import Path
from io import StringIO

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "sources" / "PDFs"
RAW_DIR = ROOT / "sources" / "raw"
METADATA_PATH = ROOT / "sources" / "metadata.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# Map each PDF to: source_id, category, title, clinical_area
PDF_MAP = [
    # ── classification ─────────────────────────────────────────────────────
    (
        "classification/Appendix 1 - Pressure Injury Staging Guide (2).pdf",
        "SRC_B01", "B",
        "Pressure Injury Staging Guide",
        "Classification — pressure injury staging",
    ),
    (
        "classification/Pressure-Ulcer-Categorisation-Tool.pdf",
        "SRC_B02", "B",
        "Pressure Ulcer Categorisation Tool",
        "Classification — pressure ulcer categories",
    ),
    (
        "classification/content_10829.pdf",
        "SRC_B03", "B",
        "Pressure Ulcer Classification: Educational Resource",
        "Classification — pressure ulcer stages",
    ),
    (
        "classification/pressure-ulcer-categorisation-poster.pdf",
        "SRC_B04", "B",
        "Pressure Ulcer Categorisation Poster",
        "Classification — quick reference",
    ),
    (
        "classification/webinar6_pu_woundassesst.pdf",
        "SRC_B05", "B",
        "Pressure Ulcer and Wound Assessment: Webinar Resource",
        "Classification and wound assessment",
    ),
    # ── nutrition ──────────────────────────────────────────────────────────
    (
        "nutrition/2025-Guideline-Nutrition-25Feb25.pdf",
        "SRC_D01", "D",
        "EPUAP 2025 Guideline: Nutrition",
        "Nutritional assessment and support",
    ),
    # ── other (aetiology + device-related) ────────────────────────────────
    (
        "other/2025-Guideline-Etiology-Dec25.pdf",
        "SRC_A01", "A",
        "EPUAP 2025 Guideline: Etiology of Pressure Injuries",
        "Aetiology and pathophysiology",
    ),
    (
        "other/2025-Guideline-DRPI-15-09-2025.pdf",
        "SRC_C01", "C",
        "EPUAP 2025 Guideline: Device-Related Pressure Injuries",
        "Device-related pressure injuries",
    ),
    # ── pain_management ────────────────────────────────────────────────────
    (
        "pain_management/content_11517.pdf",
        "SRC_E01", "E",
        "Pain Management in Pressure Ulcer Care",
        "Pain assessment and management",
    ),
    # ── patient_education ──────────────────────────────────────────────────
    (
        "patient_education/200115-Pressure-ulcer-education-5-keeping-patients-moving.pdf",
        "SRC_F01", "F",
        "Nursing Times: Pressure Ulcer Education — Keeping Patients Moving",
        "Patient education — repositioning and mobilisation",
    ),
    (
        "patient_education/MED24_ME_repostioning_WUK-WEB.pdf",
        "SRC_F02", "F",
        "Wounds UK: Best Practice Repositioning Guide",
        "Repositioning — wound care context",
    ),
    # ── prevention ─────────────────────────────────────────────────────────
    (
        "prevention/pressure-ulcers-prevention-and-management-pdf-35109760631749.pdf",
        "SRC_G01", "G",
        "NICE CG179: Pressure Ulcers — Prevention and Management",
        "UK clinical practice guideline",
    ),
    (
        "prevention/NSTPP-summary-recommendations.pdf",
        "SRC_G02", "G",
        "NHS Stop the Pressure: Summary Recommendations",
        "UK pressure ulcer prevention policy",
    ),
    # ── repositioning ──────────────────────────────────────────────────────
    (
        "repositioning/2025-Guideline-Repositioning-25-Feb-2025.pdf",
        "SRC_H01", "H",
        "EPUAP 2025 Guideline: Repositioning",
        "Repositioning and early mobilisation",
    ),
    (
        "repositioning/2025-Guideline-Seating-17Dec25.pdf",
        "SRC_H02", "H",
        "EPUAP 2025 Guideline: Seating",
        "Seating and wheelchair positioning",
    ),
    # ── risk_assessment ────────────────────────────────────────────────────
    (
        "risk_assessment/from_screening_to_full_risk_assessment_in_pressure.3.pdf",
        "SRC_I01", "I",
        "EPUAP: From Screening to Full Risk Assessment in Pressure Injury",
        "Risk assessment and screening",
    ),
    (
        "risk_assessment/Braden-teaching.pdf",
        "SRC_I02", "I",
        "Braden Scale: Teaching Guide and Scoring Instructions",
        "Risk assessment — Braden Scale",
    ),
    (
        "risk_assessment/G89_-_Guidelines_for_Pressure_ulcer_Risk_Assessment-Adapted_Waterlow_Score_P36.pdf",
        "SRC_I03", "I",
        "Guidelines for Pressure Ulcer Risk Assessment: Adapted Waterlow Score",
        "Risk assessment — Waterlow Score",
    ),
    # ── skin_assessment ────────────────────────────────────────────────────
    (
        "skin_assessment/2025-Guideline-Skin-Prevent-14Sept2025.pdf",
        "SRC_J01", "J",
        "EPUAP 2025 Guideline: Skin and Prevention",
        "Skin assessment and prevention",
    ),
    # ── support_surfaces ───────────────────────────────────────────────────
    (
        "support_surfaces/2025-Guideline-Supports-25Feb25.pdf",
        "SRC_K01", "K",
        "EPUAP 2025 Guideline: Support Surfaces",
        "Support surfaces and pressure redistribution",
    ),
    # ── wound_management ───────────────────────────────────────────────────
    (
        "wound_management/2025-Guideline-Heels-11+Sept025.pdf",
        "SRC_L01", "L",
        "EPUAP 2025 Guideline: Heel Pressure Injuries",
        "Heel pressure injury management",
    ),
]


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pdfminer."""
    output = StringIO()
    laparams = LAParams(line_margin=0.5, word_margin=0.1)
    with open(pdf_path, "rb") as f:
        extract_text_to_fp(f, output, laparams=laparams, output_type="text", codec="utf-8")
    return output.getvalue()


def clean_extracted_text(text: str) -> str:
    """Basic cleanup of PDF-extracted text."""
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines to max two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove form-feed characters
    text = text.replace("\f", "\n\n")
    # Strip leading/trailing whitespace per line
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def main():
    metadata = []
    success = 0
    failed = []

    for rel_path, source_id, category, title, clinical_area in PDF_MAP:
        pdf_path = PDF_DIR / rel_path
        txt_path = RAW_DIR / f"{source_id}.txt"

        if not pdf_path.exists():
            print(f"[MISSING] {rel_path}")
            failed.append(source_id)
            continue

        print(f"[EXTRACT] {source_id}: {pdf_path.name} ...", end=" ", flush=True)
        try:
            raw_text = extract_pdf_text(pdf_path)
            cleaned = clean_extracted_text(raw_text)

            if len(cleaned.split()) < 100:
                print(f"WARNING — only {len(cleaned.split())} words extracted (may be scanned PDF)")
            else:
                print(f"{len(cleaned.split()):,} words")

            txt_path.write_text(cleaned, encoding="utf-8")
            success += 1

            metadata.append({
                "source_id": source_id,
                "category": category,
                "title": title,
                "clinical_area": clinical_area,
                "sections_used": "Full document",
                "year": 2025 if "2025" in rel_path else 2024 if "2024" in rel_path else 2024,
                "access_date": "2026-04-29",
                "url_or_doi": "",
                "licence_notes": "Educational research use",
                "word_count": len(cleaned.split()),
                "source_file": rel_path,
            })

        except Exception as e:
            print(f"ERROR — {e}")
            failed.append(source_id)

    # Write updated metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Extracted: {success}/{len(PDF_MAP)} documents")
    print(f"Saved to:  {RAW_DIR}")
    print(f"Metadata:  {METADATA_PATH}")
    if failed:
        print(f"Failed:    {', '.join(failed)}")
    print(f"{'='*50}")
    print("\nTotal word count by category:")
    by_cat = {}
    for entry in metadata:
        by_cat.setdefault(entry["category"], []).append(entry["word_count"])
    for cat in sorted(by_cat):
        total = sum(by_cat[cat])
        print(f"  Category {cat}: {len(by_cat[cat])} docs, {total:,} words")


if __name__ == "__main__":
    main()
