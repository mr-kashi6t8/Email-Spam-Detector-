import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------------
# Load saved model & vectorizer
# -----------------------------
model = joblib.load("spam_detector.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Spam Message Detector", page_icon="📧", layout="centered")
st.title("📧 Spam Message Detector")
st.markdown("""
This app detects whether a text message is **Spam** or **Ham** (not spam).  
Enter any message below and click **Predict**.
""")

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Enter your message here:")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Transform input text
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        # Check if model supports probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_tfidf)[0]
            ham_prob, spam_prob = proba[0], proba[1]
            st.write(f"✅ Ham Probability: **{ham_prob*100:.2f}%**")
            st.write(f"⚠️ Spam Probability: **{spam_prob*100:.2f}%**")
        
        # Show prediction result
        if prediction == 1:
            st.error("⚠️ This message is **SPAM**")
        else:
            st.success("✅ This message is **HAM**")

        # -----------------------------
        # Generate WordCloud for input message
        # -----------------------------
        st.subheader("WordCloud of Input Message")
        wc = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        plt.figure(figsize=(10,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
