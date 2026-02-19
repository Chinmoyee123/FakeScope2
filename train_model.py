import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import joblib
import scipy.sparse as sp
import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")
df = df.dropna()

print("Total rows:", len(df))

# Features and label
X_text = df["cleaned_text"]
y = df["label"]

# TF-IDF
print("Applying TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(X_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training started...")

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        solver='saga'
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight='balanced',
        max_depth=10
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced'
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
}

# Train and evaluate
results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    pre = round(precision_score(y_test, y_pred) * 100, 2)
    rec = round(recall_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1 Score": f1
    })
    print(f"{name} -> Accuracy: {acc}%")

# Save results
results_df = pd.DataFrame(results)
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))
results_df.to_csv("data/model_results.csv", index=False)

# Save Logistic Regression as main model
print("\nSaving models...")
joblib.dump(models["Logistic Regression"], "models/lr_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("Models saved successfully!")