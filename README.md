# 📧 Spam Detector with Streamlit

A Machine Learning project to classify SMS/Email messages as **Spam** or **Ham (Not Spam)**.  
Built with **TF-IDF vectorization**, multiple ML models, and deployed using **Streamlit** for real-time predictions.  

---

## 📂 Dataset
- **Source**: SMS Spam Collection Dataset  
- **Features**: Message text  
- **Target**: `spam` or `ham`  

---

## ⚙️ Workflow
1. **Text Preprocessing**  
   - Lowercasing, punctuation removal, stopword removal  
   - TF-IDF Vectorization  

2. **Model Training**  
   - Tried multiple classifiers: Logistic Regression, Random Forest, XGBoost  
   - Automatic best model selection based on accuracy  

3. **Model Saving & Deployment**  
   - Saved the best model + TF-IDF vectorizer using `joblib`  
   - Deployed with **Streamlit** for real-time user input  

---

## 📊 Results
- **Best Model**: ✅ `{Linear Svm}`  
- **Accuracy**: ~95%  
- **Precision/Recall**: High for both Spam & Ham classes  
- **Confusion Matrix**: Minimal false positives/negatives  

---

## 🚀 Streamlit App
Easily test messages in real-time:  

🔗 **Live App**: [Your Streamlit App Link]  

---

## 💻 Usage
Clone repo and install requirements:
```bash
git clone https://github.com/your-username/spam-detector.git
cd spam-detector
pip install -r requirements.txt
