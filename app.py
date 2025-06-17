# -*- coding: utf-8 -*-
"""app.py - Aplikasi Streamlit Prediksi Penyakit Jantung"""

import streamlit as st
import pandas as pd
import joblib

# Load dataset dan model
url = "https://raw.githubusercontent.com/rendymalandi/last-pliss/main/Heart_Disease_Prediction.csv"
df = pd.read_csv(url)
model = joblib.load('xgboost_heart_disease_pipeline.pkl')

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Modeling", "Prediksi"])

# ====================
# Halaman 1: Dataset
# ====================
if page == "Dataset":
    st.title("📊 Dataset Penyakit Jantung")
    st.write("Cuplikan data:")
    st.dataframe(df.head())

    st.subheader("Statistik Data:")
    st.write(df.describe())

    st.subheader("Distribusi Target (Heart Disease):")
    st.bar_chart(df['Heart Disease'].value_counts())

# ====================
# Halaman 2: Modeling
# ====================
elif page == "Modeling":
    st.title("🧠 Evaluasi Model")

    try:
        from sklearn.metrics import classification_report

        # Mapping label target ke angka
        df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

        X = df.drop("Heart Disease", axis=1)
        y = df["Heart Disease"]
        y_pred = model.predict(X)

        report = classification_report(y, y_pred, output_dict=True)
        st.write("**Classification Report**:")
        st.json(report)

    except Exception as e:
        st.error(f"Error saat evaluasi model: {e}")

# ====================
# Halaman 3: Prediksi
# ====================
elif page == "Prediksi":
    st.title("🩺 Prediksi Risiko Penyakit Jantung")

    # Form input user
    age = st.number_input("Usia", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Jenis Kelamin", ["M", "F"])
    chest_pain = st.selectbox("Tipe Nyeri Dada", ["TA", "ATA", "NAP", "ASY"])
    resting_bp = st.number_input("Tekanan Darah Saat Istirahat", min_value=0, value=120)
    cholesterol = st.number_input("Kolesterol", min_value=0, value=200)
    fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1])
    resting_ecg = st.selectbox("Hasil EKG Saat Istirahat", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Detak Jantung Maksimum", min_value=0, value=150)
    exercise_angina = st.selectbox("Angina karena Latihan?", ["Y", "N"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0)
    st_slope = st.selectbox("Kemiringan ST", ["Up", "Flat", "Down"])

    # Prediksi saat tombol ditekan
    if st.button("Prediksi"):
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chest_pain],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope]
        })

        # Sesuaikan tipe data kategori
        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
        for col in categorical_cols:
            input_df[col] = input_df[col].astype("category")

        try:
            result = model.predict(input_df)[0]
            if result == 1:
                st.error("⚠️ Pasien diprediksi berisiko mengalami penyakit jantung.")
            else:
                st.success("✅ Pasien diprediksi tidak berisiko mengalami penyakit jantung.")
        except Exception as e:
            st.error("❌ Terjadi kesalahan saat prediksi.")
            st.exception(e)
