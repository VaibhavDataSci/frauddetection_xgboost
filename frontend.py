import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import requests

# ------------------------
# Load Model
# ------------------------
MODEL_PATH = "fraud_detection_model.pkl"

@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline")
        threshold = artifact.get("threshold", 0.5)
    else:
        pipeline = artifact
        threshold = 0.5
    return pipeline, threshold

pipeline, threshold = load_model()

# ------------------------
# Gemini API Query Function
# ------------------------
def query_gemini_api(question: str):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    headers = {
        "x-goog-api-key": "AIzaSyDmnum1DMOxbEmGO0bhk3R6h6WMqoMlCnQ",   # Replace with your Gemini API key
        "Content-Type": "application/json"
    }
    payload = {"prompt": question}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("answer", "Sorry, no answer returned by Gemini API.")
        else:
            return f"API request failed with status: {response.status_code}"
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# ------------------------
# Streamlit Page Setup
# ------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Fraud Detection System")
st.markdown("#### An AI-powered tool to detect fraudulent transactions & explain the reasons")

st.sidebar.header("⚙️ Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["🏠 Home", "🔍 Prediction", "📂 CSV Upload", "📊 Explanation", "💬 Fraud Assistant"]
)

# ------------------------
# Input Form
# ------------------------
def get_user_input():
    st.subheader("📝 Enter Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        type_ = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Transaction Amount", min_value=0.0, step=10.0, value=100.0)
        oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, step=10.0, value=1000.0)

    with col2:
        newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, step=10.0, value=900.0)
        oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, step=10.0, value=500.0)
        newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, step=10.0, value=600.0)

    return pd.DataFrame([{
        "type": type_,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

# ------------------------
# Feature Engineering
# ------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-9)
    return df

# ------------------------
# Prediction
# ------------------------
def make_prediction(input_df):
    proba = pipeline.predict_proba(input_df)[:, 1][0]
    label = int(proba >= threshold)
    return proba, label

# ------------------------
# SHAP Explanation
# ------------------------
def shap_explanation(input_df):
    input_df = add_engineered_features(input_df)
    try:
        X_processed = pipeline[:-1].transform(input_df)
        model = pipeline[-1]
        if hasattr(pipeline[:-1], "get_feature_names_out"):
            feature_names = pipeline[:-1].get_feature_names_out()
        else:
            feature_names = input_df.columns
    except Exception:
        X_processed = input_df.values
        model = pipeline
        feature_names = input_df.columns

    explainer = shap.KernelExplainer(model.predict_proba, X_processed)
    shap_values = explainer.shap_values(X_processed)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    st.subheader("🔍 Feature Contribution (Waterfall Plot)")
    st.set_option("deprecation.showPyplotGlobalUse", False)
    shap.plots.waterfall(shap.Explanation(values=shap_values_to_plot[0],
                                         base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                         feature_names=feature_names), show=False)
    st.pyplot(bbox_inches="tight")

    st.subheader("📄 Detailed Feature Impact")
    shap_series = pd.Series(shap_values_to_plot[0], index=feature_names).sort_values(key=abs, ascending=False)
    for feat, val in shap_series.items():
        impact = "increased" if val > 0 else "decreased"
        st.write(f"**{feat}** {impact} the probability of fraud by {abs(val):.4f}")

# ------------------------
# Pages
# ------------------------
if page == "🏠 Home":
    st.info("👋 Welcome to the **Fraud Detection System**.\n\n➡️ Use the sidebar to test predictions, upload CSVs, and get explanations.")

elif page == "🔍 Prediction":
    input_df = get_user_input()
    if st.button("🔮 Predict Transaction"):
        input_df = add_engineered_features(input_df)
        proba, label = make_prediction(input_df)
        st.metric("Fraud Probability", f"{proba:.2%}")
        if label:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Safe")

elif page == "📂 CSV Upload":
    uploaded_file = st.file_uploader("📂 Upload Transactions CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Preview of Uploaded Data", df.head())
        df = add_engineered_features(df)
        preds = pipeline.predict_proba(df)[:, 1]
        df["Fraud_Probability"] = preds
        df["Prediction"] = (preds >= threshold).astype(int)
        st.write("✅ Predictions Complete", df.head())
        st.download_button(
            "⬇️ Download Results", 
            df.to_csv(index=False), 
            "fraud_predictions.csv", 
            "text/csv"
        )

elif page == "📊 Explanation":
    st.info("ℹ️ Enter transaction details to see **why** the model predicted Fraud or Safe.")
    input_df = get_user_input()
    if st.button("🔎 Explain Prediction"):
        shap_explanation(input_df)

elif page == "💬 Fraud Assistant":
    st.write("🤖 Fraud Assistant: Ask me anything about fraud detection!")
    user_query = st.text_input("Your Question:")
    if user_query:
        answer = query_gemini_api(user_query)
        st.success(f"📝 Assistant Answer: {answer}")
