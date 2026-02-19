import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

if __name__ == "__main__":
    # Load both datasets
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    # Add labels
    fake["label"] = 0
    real["label"] = 1

    # Combine
    df = pd.concat([fake, real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep needed columns
    df = df[["title", "text", "label"]]
    df["content"] = df["title"] + " " + df["text"]
    df = df.drop(columns=["title", "text"])
    df = df.dropna()

    print("Cleaning text... please wait...")
    df["cleaned_text"] = df["content"].apply(clean_text)
    df = df.drop(columns=["content"])
    df.to_csv("data/cleaned_data.csv", index=False)

    print("Done! Total rows:", len(df))
    print("Saved as data/cleaned_data.csv")