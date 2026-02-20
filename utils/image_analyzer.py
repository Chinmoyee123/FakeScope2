import pytesseract
from PIL import Image
import re

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)


def extract_text_from_image(image):
    """
    Extract text from uploaded image using OCR
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract text using tesseract
        extracted_text = pytesseract.image_to_string(
            image,
            lang='eng',
            config='--psm 6'
        )

        # Clean extracted text
        extracted_text = extracted_text.strip()
        extracted_text = re.sub(r'\n+', ' ', extracted_text)
        extracted_text = re.sub(r'\s+', ' ', extracted_text)

        return extracted_text

    except Exception as e:
        return f"Error extracting text: {str(e)}"


def is_valid_extraction(text):
    """
    Check if extracted text is valid
    """
    # Must have at least 20 characters
    if len(text) < 20:
        return False

    # Must have at least 5 words
    if len(text.split()) < 5:
        return False

    return True