import pytesseract
from PIL import Image
import re
import os

if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = (
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )
else:
    pytesseract.pytesseract.tesseract_cmd = (
        r'/usr/bin/tesseract'
    )

def extract_text_from_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        extracted_text = pytesseract.image_to_string(
            image,
            lang='eng',
            config='--psm 6'
        )
        extracted_text = extracted_text.strip()
        extracted_text = re.sub(r'\n+', ' ', extracted_text)
        extracted_text = re.sub(r'\s+', ' ', extracted_text)
        return extracted_text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def is_valid_extraction(text):
    if len(text) < 20:
        return False
    if len(text.split()) < 5:
        return False
    return True