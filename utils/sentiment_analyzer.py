from textblob import TextBlob


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Sentiment label
    if polarity > 0.1:
        sentiment_label = "Positive 😊"
    elif polarity < -0.1:
        sentiment_label = "Negative 😠"
    else:
        sentiment_label = "Neutral 😐"

    # Calculate sentiment score for hybrid
    # Real news tends to be neutral and objective
    # Fake news tends to be very negative/positive and subjective

    sentiment_score = 50  # start neutral

    # Very high subjectivity = likely fake
    if subjectivity > 0.7:
        sentiment_score -= 20
    elif subjectivity < 0.4:
        sentiment_score += 20

    # Very extreme polarity = likely fake
    if polarity > 0.5 or polarity < -0.5:
        sentiment_score -= 15
    else:
        sentiment_score += 15

    # Clamp between 0 and 100
    sentiment_score = max(0, min(100, sentiment_score))

    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "sentiment_label": sentiment_label,
        "sentiment_score": round(sentiment_score, 2)
    }