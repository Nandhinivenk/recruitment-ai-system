import sys
from pdf_reader import extract_text_from_pdf
from ocr_reader import extract_text_from_scanned_pdf


def test_pdf_parsing(pdf_path):
    print("\n📄 Testing PDF:", pdf_path)
    print("-" * 50)

    text = extract_text_from_pdf(pdf_path)

    print("Characters extracted (pdfplumber):", len(text))
    print("\nPreview:\n", text[:800])

    if len(text.strip()) < 100:
        print("\n⚠ Low text detected → Switching to OCR...\n")
        text = extract_text_from_scanned_pdf(pdf_path)
        print("Characters extracted (OCR):", len(text))
        print("\nOCR Preview:\n", text[:])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_parsing.py <path_to_pdf>")
        sys.exit(1)

    test_pdf_parsing(sys.argv[1])
