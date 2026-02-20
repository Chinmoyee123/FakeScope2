# Fake news indicator words
FAKE_KEYWORDS = [
    # Conspiracy words
    "conspiracy", "hoax", "coverup", "cover-up",
    "they dont want you to know", "hidden truth",
    "wake up", "sheeple", "illuminati", "new world order",
    "deep state", "false flag", "staged", "fake",
    "mainstream media lying", "government hiding",
    "secretly", "secret plan", "secretly planning",
    "they are hiding", "hidden agenda",

    # Urgency/panic words
    "immediately", "right now", "before its too late",
    "too late", "withdraw immediately",
    "act now", "hurry", "urgent warning",
    "last chance", "do it now", "before they",
    "shut down", "collapse", "crash",

    # Sensational words
    "shocking", "bombshell", "explosive", "breaking",
    "unbelievable", "jaw dropping", "mind blowing",
    "you wont believe", "share before deleted",
    "banned video", "censored", "they are hiding",
    "single tweet", "village scientist",
    "one simple trick",

    # Health misinformation
    "miracle cure", "doctors dont want",
    "big pharma", "natural cure", "cancer cure",
    "vaccine kills", "microchip", "5g causes",
    "detox", "cleanse cures", "tracking chips",
    "mass surveillance", "vaccine contain",
    "chips used", "tracking device",

    # Financial panic
    "banks will close", "bank shutdown",
    "withdraw all", "banks next month",
    "financial collapse", "market crash",
    "banks shutting", "pull out money",
    "government shutdown banks",
    "all banks", "shut down all banks",

    # Clickbait
    "what happened next", "this is why",
    "the truth about", "exposed", "secret revealed",
    "what they dont tell", "read before deleted",
    "share before", "delete this",
    "proven that", "single person proved",
    "one scientist", "village scientist",
    "one tweet proves",
    
     # Misleading scientific claims
    "ayurvedic doctor", "traditional medicine in space",
    "selects indian", "alternative medicine nasa",
    "nasa selects", "explore role",
    "homeopathy in space", "spiritual healing",
    "ancient remedy", "traditional healer",
    "witch doctor", "herbal cure in space",
    "yoga cures cancer", "meditation cures",
    "ayurveda cures", "natural healing space",

    # Fake authority claims
    "nasa confirms flat", "nasa admits",
    "nasa hides", "nasa secret",
    "who admits", "cdc admits",
    "government admits", "scientists admit",
    "experts admit", "doctor admits",
    "professor admits", "researcher admits",

    # Implausible combinations
    "village doctor nasa", "local scientist nasa",
    "indian doctor space", "traditional doctor space",
    "spiritual advisor nasa", "astrologer confirms",
    "psychic predicts", "fortune teller"
]

# Real news indicator words
REAL_KEYWORDS = [
    # Attribution words
    "according to", "reported by", "confirmed by",
    "stated by", "announced by", "said in a statement",
    "press conference", "official statement",
    "peer reviewed", "published in journal",
    "research paper", "clinical trial",

    # News agency references
    "reuters", "associated press", "ap news",
    "bbc", "cnn reported", "new york times",
    "washington post", "the guardian",
    "bloomberg", "financial times",

    # Official sources
    "government official", "spokesperson said",
    "minister said", "president said",
    "official data", "statistics show",
    "research shows", "study finds",
    "scientists say", "experts say",
    "poll shows", "survey finds",
    "central bank", "federal reserve",
    "world health organization", "who confirmed",
    "cdc confirmed", "fda approved",

    # Formal language
    "percent", "billion", "million",
    "committee", "legislation", "parliament",
    "federal", "national", "international",
    "department", "ministry", "agency",
    "quarterly report", "annual report",
    "data shows", "evidence suggests",
    "independent research", "multiple sources"
]

def get_keyword_score(text):
    text_lower = text.lower()

    # Count fake keywords found
    fake_count = 0
    fake_found = []
    for word in FAKE_KEYWORDS:
        if word in text_lower:
            fake_count += 1
            fake_found.append(word)

    # Count real keywords found
    real_count = 0
    real_found = []
    for word in REAL_KEYWORDS:
        if word in text_lower:
            real_count += 1
            real_found.append(word)

    # Calculate score
    total = fake_count + real_count
    if total == 0:
        keyword_score = 50  # neutral
    else:
        real_score = (real_count / total) * 100
        fake_score = (fake_count / total) * 100
        keyword_score = real_score

    return {
        "keyword_score": round(keyword_score, 2),
        "fake_keywords_found": fake_found,
        "real_keywords_found": real_found,
        "fake_count": fake_count,
        "real_count": real_count
    }