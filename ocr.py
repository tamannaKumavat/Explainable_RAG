from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
import tempfile
import os

def extract_text_from_pdf(pdf_file_path):
    text = ""
    # Use 'with' to ensure file is properly closed
    with open(pdf_file_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except Exception:
                # If extract_text() fails, fallback to OCR
                with tempfile.TemporaryDirectory() as path:
                    images = convert_from_path(pdf_file_path, output_folder=path)
                    for image in images:
                        text += pytesseract.image_to_string(image)
                break  # No need to try OCR for every page, do it once for whole PDF
    return text