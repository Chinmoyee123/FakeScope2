# FakeScope 2.0 🔍
### Advanced Fake News Detection using Hybrid AI

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![ML](https://img.shields.io/badge/ML-Scikit--learn-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 About
FakeScope 2.0 is an advanced hybrid fake news detection 
system that combines Machine Learning, Sentiment Analysis, 
Keyword Detection and Writing Style Analysis to detect 
fake news with high accuracy.

---

## 🚀 Features
- ✅ Hybrid Detection System (4 components)
- ✅ Explains WHY news is fake or real
- ✅ Sentiment Analysis (Polarity + Subjectivity)
- ✅ Keyword Detection System
- ✅ Writing Style Analysis
- ✅ 6 ML Models Compared
- ✅ Beautiful Interactive Dashboard
- ✅ Radar Chart + Gauge Chart visualization

---

## 🛠️ Tech Stack
- **Language:** Python 3.11
- **Framework:** Streamlit
- **ML:** Scikit-learn, XGBoost
- **NLP:** NLTK, TextBlob
- **Visualization:** Plotly
- **Dataset:** Bisaillon Fake News Dataset

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 99.30% | 99.07% | 99.46% | 99.27% |
| Decision Tree | 99.62% | 99.44% | 99.77% | 99.60% |
| Random Forest | 99.59% | 99.53% | 99.60% | 99.57% |
| AdaBoost | 99.60% | 99.35% | 99.81% | 99.58% |
| KNN | 92.99% | 89.53% | 96.61% | 92.94% |
| XGBoost | 99.68% | 99.49% | 99.84% | 99.66% |

---

## 🔍 How It Works
```
Input News Article
      ↓
1. Text Preprocessing
      ↓
2. Sentiment Analysis (15% weight)
      ↓
3. Keyword Detection (40% weight)
      ↓
4. Writing Style Analysis (20% weight)
      ↓
5. ML Model - Logistic Regression (25% weight)
      ↓
Hybrid Score Calculation
      ↓
Final Result (Real/Fake + Explanation)
```

---

## 📁 Project Structure
```
FakeScope2/
│
├── app.py                  ← Streamlit web app
├── hybrid_scorer.py        ← Hybrid scoring system
├── preprocess.py           ← Data preprocessing
├── train_model.py          ← ML model training
├── requirements.txt        ← Dependencies
├── README.md               ← Documentation
│
├── data/
│   ├── Fake.csv            ← Fake news dataset
│   ├── True.csv            ← Real news dataset
│   ├── cleaned_data.csv    ← Preprocessed data
│   └── model_results.csv   ← Model comparison
│
├── models/
│   ├── lr_model.pkl        ← Saved LR model
│   └── tfidf_vectorizer.pkl← Saved TF-IDF
│
└── utils/
    ├── keywords.py         ← Keyword rules
    ├── sentiment_analyzer.py← Sentiment analysis
    └── style_analyzer.py   ← Style analysis
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/FakeScope2.git
cd FakeScope2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download Fake.csv and True.csv from:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Place them in the data/ folder

### 4. Preprocess and Train
```bash
python preprocess.py
python train_model.py
```

### 5. Run the App
```bash
streamlit run app.py
```

---

## 📈 Hybrid Scoring System

| Component | Weight | Description |
|-----------|--------|-------------|
| ML Model | 25% | Logistic Regression on TF-IDF |
| Sentiment | 15% | Polarity + Subjectivity |
| Keywords | 40% | Fake/Real word detection |
| Style | 20% | Writing style patterns |
```
---

## ⚠️ Limitations
- Model trained on English news only
- Dataset bias towards specific writing styles
- May not detect sophisticated fake news
- Performance depends on article length

---

## 🔮 Future Work
- Add multilingual support
- Add URL/source verification
- Add deep learning models
- Deploy on cloud platform

---

## 📄 License
MIT License