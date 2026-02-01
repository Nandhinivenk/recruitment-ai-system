import pytesseract
from pdf2image import convert_from_path


def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    text = ""

    images = convert_from_path(
        pdf_path,
        dpi=300
        # poppler_path=POPPLER_PATH
    )

    for img in images:
        page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"

    return text
