import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from hybrid_scorer import analyze_news

# Page config
st.set_page_config(
    page_title="FakeScope 2.0",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        width: 100%;
        background-color: #1a1a2e;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 18px;
    }
    .fake-box {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid red;
        font-size: 24px;
        font-weight: bold;
        color: red;
    }
    .real-box {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid green;
        font-size: 24px;
        font-weight: bold;
        color: green;
    }
    .score-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image(
    "https://img.icons8.com/color/96/000000/news.png",
    width=80
)
st.sidebar.title("🔍 FakeScope 2.0")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔎 Analyze News",
    "📊 Model Results",
    "📖 How It Works",
    "👥 About"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
### How Scoring Works
- 🤖 ML Model: 25%
- 💬 Sentiment: 15%
- 🔑 Keywords: 40%
- ✍️ Style: 20%
""")

# ===== HOME PAGE =====
if page == "🏠 Home":
    st.title("🔍 FakeScope 2.0")
    st.subheader("Advanced Fake News Detection using Hybrid AI")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("📰 44,000+\nNews Trained")
    with col2:
        st.success("🤖 6 ML Models\nCompared")
    with col3:
        st.warning("🎯 99%+\nAccuracy")
    with col4:
        st.error("🔑 Hybrid\nDetection")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What is FakeScope 2.0?
        FakeScope 2.0 is an advanced hybrid fake news
        detection system that combines:
        - 🤖 **Machine Learning** — TF-IDF + 
        Logistic Regression
        - 💬 **Sentiment Analysis** — Polarity + 
        Subjectivity
        - 🔑 **Keyword Detection** — Fake/Real 
        word patterns
        - ✍️ **Writing Style Analysis** — CAPS, 
        punctuation, length
        """)

    with col2:
        st.markdown("""
        ### Why FakeScope 2.0?
        Unlike traditional fake news detectors that
        rely only on ML models, FakeScope 2.0:
        - Uses **4 different signals** combined
        - **Explains WHY** it thinks news is fake
        - Works on **real world news** articles
        - Shows **detailed analysis** breakdown
        - Provides **transparent scoring** system
        """)

    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.markdown("Click on **🔎 Analyze News** in the sidebar to start analyzing news articles!")

# ===== ANALYZE PAGE =====
elif page == "🔎 Analyze News":
    st.title("🔎 Analyze News Article")
    st.markdown("---")

    news_input = st.text_area(
        "Paste your news article here:",
        height=250,
        placeholder="Paste any news article here and click Analyze..."
    )

    if st.button("🔍 ANALYZE NEWS"):
        if news_input.strip() == "":
            st.warning("Please enter some news text!")
        else:
            with st.spinner("Analyzing news using Hybrid AI..."):
                results = analyze_news(news_input)

            st.markdown("---")

            # Main result
            if results["prediction"] == "REAL":
                st.markdown(
                    f'<div class="real-box">'
                    f'✅ REAL NEWS — '
                    f'Confidence: {results["confidence"]}%'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="fake-box">'
                    f'❌ FAKE NEWS — '
                    f'Confidence: {results["confidence"]}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # Score breakdown
            st.subheader("📊 Score Breakdown")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🤖 ML Score",
                    f"{results['ml_score']}%",
                    help="Machine Learning model score"
                )
            with col2:
                st.metric(
                    "💬 Sentiment Score",
                    f"{results['sentiment_score']}%",
                    help="Sentiment analysis score"
                )
            with col3:
                st.metric(
                    "🔑 Keyword Score",
                    f"{results['keyword_score']}%",
                    help="Keyword detection score"
                )
            with col4:
                st.metric(
                    "✍️ Style Score",
                    f"{results['style_score']}%",
                    help="Writing style score"
                )

            st.markdown("---")

            # Hybrid score gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results["hybrid_score"],
                title={"text": "Hybrid Score (>50 = Real)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": "green"
                        if results["prediction"] == "REAL"
                        else "red"
                    },
                    "steps": [
                        {"range": [0, 50],
                         "color": "#ffcccc"},
                        {"range": [50, 100],
                         "color": "#ccffcc"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Score radar chart
            categories = [
                'ML Score',
                'Sentiment Score',
                'Keyword Score',
                'Style Score'
            ]
            values = [
                results['ml_score'],
                results['sentiment_score'],
                results['keyword_score'],
                results['style_score']
            ]

            fig2 = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line_color='green'
                if results["prediction"] == "REAL"
                else 'red'
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(
                    visible=True, range=[0, 100]
                )),
                title="Score Radar Chart"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            # Explanation
            st.subheader("🔍 Why This Result?")
            for exp in results["explanation"]:
                st.markdown(f"- {exp}")

            st.markdown("---")

            # Sentiment details
            st.subheader("💬 Sentiment Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Sentiment",
                    results["sentiment"]["sentiment_label"]
                )
            with col2:
                st.metric(
                    "Polarity",
                    results["sentiment"]["polarity"]
                )
            with col3:
                st.metric(
                    "Subjectivity",
                    results["sentiment"]["subjectivity"]
                )

            st.markdown("---")

            # Writing style details
            st.subheader("✍️ Writing Style Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Words",
                    results["style"]["num_words"]
                )
            with col2:
                st.metric(
                    "Sentences",
                    results["style"]["num_sentences"]
                )
            with col3:
                st.metric(
                    "Exclamations",
                    results["style"]["exclamation_count"]
                )
            with col4:
                st.metric(
                    "CAPS Words",
                    results["style"]["caps_words_count"]
                )

            st.markdown("---")

            # Keywords found
            st.subheader("🔑 Keywords Found")
            col1, col2 = st.columns(2)
            with col1:
                st.error(
                    f"⚠️ Suspicious Words Found: "
                    f"{results['keywords']['fake_count']}\n\n"
                    + (", ".join(
                        results['keywords']['fake_keywords_found']
                    ) if results['keywords']['fake_keywords_found']
                      else "None found ✅")
                )
            with col2:
                st.success(
                    f"✅ Credible Words Found: "
                    f"{results['keywords']['real_count']}\n\n"
                    + (", ".join(
                        results['keywords']['real_keywords_found']
                    ) if results['keywords']['real_keywords_found']
                      else "None found ⚠️")
                )

# ===== MODEL RESULTS PAGE =====
elif page == "📊 Model Results":
    st.title("📊 Model Comparison Results")
    st.markdown("---")

    try:
        results_df = pd.read_csv("data/model_results.csv")

        best = results_df.loc[
            results_df["Accuracy"].idxmax()
        ]
        st.success(
            f"🏆 Best Model: **{best['Model']}** "
            f"with Accuracy: **{best['Accuracy']}%**"
        )

        st.markdown("---")
        st.subheader("All Models Performance")
        st.dataframe(results_df, use_container_width=True)

        st.markdown("---")

        fig = px.bar(
            results_df,
            x="Model", y="Accuracy",
            color="Model",
            title="Accuracy Comparison",
            text="Accuracy"
        )
        fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside'
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            results_df.melt(
                id_vars="Model",
                value_vars=[
                    "Accuracy", "Precision",
                    "Recall", "F1 Score"
                ]
            ),
            x="Model", y="value",
            color="variable",
            barmode="group",
            title="All Metrics Comparison"
        )
        st.plotly_chart(fig2, use_container_width=True)

    except:
        st.error(
            "model_results.csv not found! "
            "Please run train_model.py first."
        )

# ===== HOW IT WORKS PAGE =====
elif page == "📖 How It Works":
    st.title("📖 How FakeScope 2.0 Works")
    st.markdown("---")

    st.markdown("""
    ### Hybrid Detection System

    FakeScope 2.0 uses 4 components to detect fake news:

    ---

    #### 1. 🤖 Machine Learning Model (35% weight)
    - Text cleaned and preprocessed
    - TF-IDF converts text to numbers
    - Logistic Regression predicts Real/Fake
    - Trained on 44,000+ news articles

    ---

    #### 2. 💬 Sentiment Analysis (20% weight)
    - Polarity score calculated (-1 to +1)
    - Subjectivity score calculated (0 to 1)
    - Real news tends to be neutral and objective
    - Fake news tends to be extreme and subjective

    ---

    #### 3. 🔑 Keyword Detection (25% weight)
    - Checks for fake news trigger words
    - Checks for credible news indicator words
    - Examples of fake words: conspiracy, hoax, miracle cure
    - Examples of real words: according to, confirmed by, reuters

    ---

    #### 4. ✍️ Writing Style Analysis (20% weight)
    - Counts CAPS words
    - Counts exclamation marks
    - Checks sentence length
    - Checks article length
    - Real news tends to be formal and detailed

    ---

    ### Final Score Calculation
```
    Hybrid Score =
        ML Score × 25% +
        Sentiment Score × 15% +
        Keyword Score × 40% +
        Style Score × 20%

    If Hybrid Score >= 50 → REAL NEWS
    If Hybrid Score < 50  → FAKE NEWS
```
    """)

# ===== ABOUT PAGE =====
elif page == "👥 About":
    st.title("👥 About FakeScope 2.0")
    st.markdown("---")

    st.markdown("""
    ### Project Details
    - **Project Name:** FakeScope 2.0
    - **Type:** Major Project
    - **Domain:** Natural Language Processing
    - **Topic:** Sentiment Analysis on Fake News Detection

    ---

    ### Technologies Used
    - **Language:** Python
    - **Framework:** Streamlit
    - **ML Library:** Scikit-learn
    - **NLP:** NLTK, TextBlob
    - **Visualization:** Plotly
    - **Dataset:** Bisaillon Fake News Dataset (44,898 articles)

    ---

     ### Models Compared

    | Model | Accuracy | Precision | Recall | F1 Score |
    |-------|----------|-----------|--------|----------|
    | Logistic Regression | 99.30% | 99.07% | 99.46% | 99.27% |
    | Decision Tree | 99.62% | 99.44% | 99.77% | 99.60% |
    | Random Forest | 99.59% | 99.53% | 99.60% | 99.57% |
    | AdaBoost | 99.60% | 99.35% | 99.81% | 99.58% |
    | KNN | 92.99% | 89.53% | 96.61% | 92.94% |
    | XGBoost | 99.68% | 99.49% | 99.84% | 99.66% |

    🏆 Best Model: XGBoost with 99.68% Accuracy

    ---

    ### Detection Components
    - Machine Learning Model
    - Sentiment Analysis
    - Keyword Detection
    - Writing Style Analysis
    """)