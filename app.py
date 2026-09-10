import re
import joblib
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Support Ticket Intelligence", layout="centered")


@st.cache_resource
def ensure_nltk_data():
    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


ensure_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
negation_words = {
    "not", "no", "nor", "never", "n't", "cannot", "won",
    "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't",
    "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't",
    "shouldn't", "wasn't", "weren't", "won't", "wouldn't", "out",
}
stop_words = stop_words - negation_words


def nlpfunc(text):
    text = str(text).lower()
    text = re.sub(r"\\n+", " ", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


@st.cache_resource
def load_artifacts():
    return {
        "pipe_q": joblib.load("final_pipe_q.pkl"),
        "pipe_p": joblib.load("final_pipe_p.pkl"),
        "queue_encode": joblib.load("queue_encode.pkl"),
        "priority_encode": joblib.load("priority_encode.pkl"),
        "search_vectorizer": joblib.load("search_vectorizer.pkl"),
        "search_matrix": joblib.load("search_matrix.pkl"),
        "search_df": joblib.load("search_df.pkl"),
    }


artifacts = load_artifacts()


def find_similar_tickets(new_text, top_n=5):
    cleaned = nlpfunc(new_text)
    query_vec = artifacts["search_vectorizer"].transform([cleaned])
    similarities = cosine_similarity(query_vec, artifacts["search_matrix"]).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    results = artifacts["search_df"].iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]
    return results


st.title("Support Ticket Intelligence")
st.caption(
    "Predicts department (queue) and urgency (priority) for a new support ticket, "
    "and surfaces similar past tickets with how they were resolved."
)

with st.form("ticket_form"):
    subject = st.text_input("Subject")
    body = st.text_area("Describe the issue", height=150)
    ticket_type = st.selectbox("Type", ["Incident", "Request", "Problem", "Change"])
    submitted = st.form_submit_button("Analyze ticket")

if submitted:
    if not subject.strip() and not body.strip():
        st.warning("Please enter a subject or description.")
    else:
        subject_clean = nlpfunc(subject)
        body_clean = nlpfunc(body)
        combine_text = subject_clean + " " + body_clean

        input_df = pd.DataFrame({"combine_text": [combine_text], "type": [ticket_type]})

        pred_queue = artifacts["queue_encode"].inverse_transform(
            artifacts["pipe_q"].predict(input_df)
        )[0]
        pred_priority = artifacts["priority_encode"].inverse_transform(
            artifacts["pipe_p"].predict(input_df)
        )[0]

        col1, col2 = st.columns(2)
        col1.metric("Predicted queue", pred_queue)
        col2.metric("Predicted priority", pred_priority)

        st.subheader("Similar past tickets")
        similar = find_similar_tickets(subject + " " + body, top_n=5)
        for _, row in similar.iterrows():
            with st.expander(f"{row['queue']} / {row['priority']}  (similarity {row['similarity']:.2f})"):
                st.write("**Ticket:**", str(row["body"])[:400])
                st.write("**Resolution:**", str(row["answer"])[:400])

st.divider()
st.caption("TF-IDF + Random Forest, trained from scratch. Full write-up on GitHub.")
