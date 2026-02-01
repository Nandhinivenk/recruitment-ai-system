import re

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)               # remove emails
    text = re.sub(r'\+?\d[\d\s\-]{8,}', '', text)     # remove phone numbers
    text = re.sub(r'[^a-z0-9\s\.\,\-\+]', ' ', text)  # remove symbols
    text = re.sub(r'\s+', ' ', text)                  # normalize spaces
    return text.strip()
