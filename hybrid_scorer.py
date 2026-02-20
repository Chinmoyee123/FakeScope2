from utils.keywords import get_keyword_score
from utils.style_analyzer import (analyze_writing_style,
                                   check_implausibility)
from utils.sentiment_analyzer import analyze_sentiment
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load model and vectorizer
model = joblib.load("models/lr_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def get_ml_score(text):
    cleaned = clean_text(text)
    tfidf_input = tfidf.transform([cleaned])
    probability = model.predict_proba(tfidf_input)[0]
    real_prob = round(probability[1] * 100, 2)
    fake_prob = round(probability[0] * 100, 2)
    return {
        "ml_score": real_prob,
        "fake_prob": fake_prob,
        "real_prob": real_prob
    }


def analyze_news(text):
    # Get all scores
    ml_results = get_ml_score(text)
    sentiment_results = analyze_sentiment(text)
    keyword_results = get_keyword_score(text)
    style_results = analyze_writing_style(text)
    implausibility_results = check_implausibility(text)

    # Hybrid score calculation
    ml_weight = 0.25
    sentiment_weight = 0.15
    keyword_weight = 0.40
    style_weight = 0.20

    # Calculate weighted hybrid score
    hybrid_score = (
        ml_results["ml_score"] * ml_weight +
        sentiment_results["sentiment_score"] * sentiment_weight +
        keyword_results["keyword_score"] * keyword_weight +
        style_results["style_score"] * style_weight
    )

    # Apply implausibility penalty
    if implausibility_results["implausibility_score"] > 0:
        hybrid_score = hybrid_score - (
            implausibility_results["implausibility_score"] * 0.5
        )
        hybrid_score = max(0, hybrid_score)

    hybrid_score = round(hybrid_score, 2)

    # Final prediction
    if hybrid_score >= 50:
        prediction = "REAL"
        confidence = hybrid_score
    else:
        prediction = "FAKE"
        confidence = round(100 - hybrid_score, 2)

    # Generate explanation
    explanation = []

    if keyword_results["fake_count"] > 0:
        explanation.append(
            f"⚠️ Found {keyword_results['fake_count']} "
            f"suspicious words: "
            f"{', '.join(keyword_results['fake_keywords_found'][:3])}"
        )

    if keyword_results["real_count"] > 0:
        explanation.append(
            f"✅ Found {keyword_results['real_count']} "
            f"credible words: "
            f"{', '.join(keyword_results['real_keywords_found'][:3])}"
        )

    if style_results["exclamation_count"] > 2:
        explanation.append(
            f"⚠️ Too many exclamation marks "
            f"({style_results['exclamation_count']}) — "
            f"sensational writing style"
        )

    if style_results["caps_words_count"] > 3:
        explanation.append(
            f"⚠️ Too many CAPS words "
            f"({style_results['caps_words_count']}) — "
            f"aggressive writing style"
        )

    if sentiment_results["subjectivity"] > 0.7:
        explanation.append(
            f"⚠️ Very high subjectivity "
            f"({sentiment_results['subjectivity']}) — "
            f"opinion based not fact based"
        )

    if style_results["num_words"] > 100:
        explanation.append(
            f"✅ Detailed article "
            f"({style_results['num_words']} words) — "
            f"real news tends to be detailed"
        )
        
    if implausibility_results["implausible_found"]:
        explanation.append(
            f"⚠️ Implausible combination detected: "
            f"{', '.join(implausibility_results['implausible_found'])}"
            f" — this combination is highly unlikely to be real"
        )    

    if not explanation:
        if prediction == "REAL":
            explanation.append(
                "✅ Writing style and language "
                "patterns suggest credible news"
            )
        else:
            explanation.append(
                "⚠️ Writing style and language "
                "patterns suggest suspicious news"
            )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "hybrid_score": hybrid_score,
        "ml_score": ml_results["ml_score"],
        "sentiment_score": sentiment_results["sentiment_score"],
        "keyword_score": keyword_results["keyword_score"],
        "style_score": style_results["style_score"],
        "sentiment": sentiment_results,
        "keywords": keyword_results,
        "style": style_results,
        "explanation": explanation
    }