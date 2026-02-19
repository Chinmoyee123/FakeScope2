import re


def analyze_writing_style(text):
    # Count various style features
    
    # All caps words (shouting)
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    caps_count = len(caps_words)

    # Exclamation marks
    exclamation_count = text.count('!')

    # Question marks
    question_count = text.count('?')

    # Average word length
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

    # Average sentence length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    avg_sentence_length = sum(len(s.split()) 
                             for s in sentences) / len(sentences) if sentences else 0

    # Number of sentences
    num_sentences = len(sentences)

    # Number of words
    num_words = len(words)

    # URLs count
    urls = re.findall(r'http\S+|www\S+', text)
    url_count = len(urls)

    # Calculate style score
    # Fake news tends to have:
    # - More caps words
    # - More exclamation marks
    # - Shorter sentences
    # - More question marks

    fake_style_signals = 0
    real_style_signals = 0

    # Check caps
    if caps_count > 3:
        fake_style_signals += 2
    else:
        real_style_signals += 2

    # Check exclamation
    if exclamation_count > 2:
        fake_style_signals += 2
    else:
        real_style_signals += 2

    # Check sentence length
    if avg_sentence_length < 10:
        fake_style_signals += 1
    elif avg_sentence_length > 20:
        real_style_signals += 2
    else:
        real_style_signals += 1

    # Check word length
    if avg_word_length > 5:
        real_style_signals += 2
    else:
        fake_style_signals += 1

    # Check question marks
    if question_count > 3:
        fake_style_signals += 1
    else:
        real_style_signals += 1

    # Check text length
    if num_words > 100:
        real_style_signals += 2
    else:
        fake_style_signals += 1

    # Calculate final style score
    total_signals = fake_style_signals + real_style_signals
    if total_signals == 0:
        style_score = 50
    else:
        style_score = (real_style_signals / total_signals) * 100

    return {
        "style_score": round(style_score, 2),
        "caps_words_count": caps_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "num_words": num_words,
        "num_sentences": num_sentences,
        "url_count": url_count,
        "caps_words": caps_words[:5]
    }