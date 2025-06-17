# -*- coding: utf-8 -*-
"""app.ipynb

Streamlit app untuk prediksi penyakit jantung
"""

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

# Halaman 1 - Dataset
if page == "Dataset":
    st.title("📊 Dataset Penyakit Jantung")
    st.write("Berikut adalah cuplikan data yang digunakan:")
    st.dataframe(df.head())

    st.subheader("Informasi Dataset:")
    st.write(df.describe())

    st.subheader("Distribusi Target:")
    st.bar_chart(df['Heart Disease'].value_counts())

# Halaman 2 - Modeling
elif page == "Modeling":
    st.title("🧠 Evaluasi Model")
    st.write("Model ini telah dilatih sebelumnya menggunakan algoritma machine learning.")
    st.markdown("- 📁 Model: xgboost_heart_disease_pipeline.pkl")
    st.markdown("- ✅ Fitur: Berdasarkan data pasien (usia, tekanan darah, kolesterol, dll)")

    try:
        from sklearn.metrics import classification_report
        X = df.drop("Heart Disease", axis=1)
        y = df["Heart Disease"]
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        st.write("**Classification Report**:")
        st.json(report)
    except Exception as e:
        st.warning(f"Gagal menghitung metrik. Error: {e}")

# Halaman 3 - Prediksi
elif page == "Prediksi":
    st.title("🩺 Prediksi Penyakit Jantung")

    # Form input
    age = st.number_input("Usia", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Jenis Kelamin", ["M", "F"])
    chest_pain = st.selectbox("Tipe Nyeri Dada", ["TA", "ATA", "NAP", "ASY"])
    resting_bp = st.number_input("Tekanan Darah Saat Istirahat", min_value=0, value=120)
    cholesterol = st.number_input("Kolesterol", min_value=0, value=200)
    fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1])
    resting_ecg = st.selectbox("Elektrokardiografi Saat Istirahat", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Detak Jantung Maksimum", min_value=0, value=150)
    exercise_angina = st.selectbox("Angina karena Latihan?", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", value=1.0)
    st_slope = st.selectbox("Kemiringan ST", ["Up", "Flat", "Down"])

    # Tombol prediksi
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

        # Konversi kolom kategorikal sesuai pipeline model
        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
        for col in categorical_cols:
            input_df[col] = input_df[col].astype("category")

        # Prediksi
        try:
            result = model.predict(input_df)[0]
            if result == 1:
                st.error("⚠️ Pasien berisiko mengalami penyakit jantung.")
            else:
                st.success("✅ Pasien tidak berisiko mengalami penyakit jantung.")
        except Exception as e:
            st.error("❌ Terjadi kesalahan saat prediksi.")
            st.exception(e)
