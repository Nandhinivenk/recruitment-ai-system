import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from embedder import SBERTEmbedder
from preprocess import preprocess_text
from pdf_reader import extract_text_from_pdf
from ocr_reader import extract_text_from_scanned_pdf


# ---------------- CONFIG ----------------
MATCH_THRESHOLD = 55.0  # percentage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JD_PATH = os.path.join(
    BASE_DIR, "..", "data", "job_descriptions", "jd_backend_dev.txt"
)

RESUME_DIR = os.path.join(
    BASE_DIR, "..", "data", "resumes"   # 👈 SINGLE UNIFIED FOLDER
)
# ----------------------------------------


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_resume_text(file_path):
    """
    Automatically detects resume type and extracts text.
    """
    if file_path.endswith(".txt"):
        return load_text(file_path)

    elif file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)

        # Fallback to OCR for scanned/template PDFs
        if len(text.strip()) < 100:
            text = extract_text_from_scanned_pdf(file_path)

        return text

    else:
        return ""


def classify_tier(score_percent):
    if score_percent >= 70:
        return "Strong Match"
    elif score_percent >= MATCH_THRESHOLD:
        return "Potential Match"
    else:
        return "Weak Match"


def main():
    embedder = SBERTEmbedder()

    # Load and preprocess JD
    jd_text = preprocess_text(load_text(JD_PATH))
    jd_embedding = embedder.embed(jd_text)

    results = []

    for resume_file in os.listdir(RESUME_DIR):
        resume_path = os.path.join(RESUME_DIR, resume_file)

        if not os.path.isfile(resume_path):
            continue

        raw_text = load_resume_text(resume_path)

        if len(raw_text.strip()) < 50:
            print(f"⚠ Skipping empty resume: {resume_file}")
            continue

        resume_text = preprocess_text(raw_text)
        resume_embedding = embedder.embed(resume_text)

        similarity = cosine_similarity(
            [jd_embedding], [resume_embedding]
        )[0][0]

        score_percent = round(similarity * 100, 2)
        tier = classify_tier(score_percent)
        decision = "Selected" if score_percent >= MATCH_THRESHOLD else "Rejected"

        results.append({
            "Resume": resume_file,
            "Match_Score (%)": score_percent,
            "Tier": tier,
            "Decision": decision
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Match_Score (%)", ascending=False)

    print("\nSemantic Matching Results:\n")
    print(df.to_string(index=False))

    # Save output
    output_path = os.path.join(BASE_DIR, "..", "outputs", "semantic_ranking.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
