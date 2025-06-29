# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load("xgboost_heart_disease_pipeline.pkl")

# Sidebar navigasi
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Modeling", "Prediksi"])

# Halaman Dataset
if page == "Dataset":
    st.title("📊 Dataset Penyakit Jantung")
    url = "https://raw.githubusercontent.com/rendymalandi/last-pliss/main/Heart_Disease_Prediction.csv"
    df = pd.read_csv(url)
    st.dataframe(df.head())
    st.subheader("Distribusi Target")
    st.bar_chart(df['Heart Disease'].value_counts())

# Halaman Modeling
elif page == "Modeling":
    st.title("🧠 Evaluasi Model")
    st.markdown("Model: `xgboost_heart_disease_pipeline.pkl`")

    try:
        url = "https://raw.githubusercontent.com/rendymalandi/last-pliss/main/Heart_Disease_Prediction.csv"
        df = pd.read_csv(url)
        from sklearn.metrics import classification_report, accuracy_score

        X = df.drop("Heart Disease", axis=1)
        y = df["Heart Disease"]

        # Ubah kategorikal jadi kategori
        categorical_cols = [
            "Sex", "Chest pain type", "EKG results", "Exercise angina",
            "Slope of ST", "Thallium"
        ]
        for col in categorical_cols:
            X[col] = X[col].astype("category")

        # Encode label jika perlu
        if y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        st.metric("🎯 Akurasi", f"{acc:.2%}")
        st.subheader("Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.json(report)

    except Exception as e:
        st.warning(f"Error saat evaluasi model: {e}")

# Halaman Prediksi
elif page == "Prediksi":
    st.title("🩺 Prediksi Risiko Penyakit Jantung")

    # Input pengguna
    age = st.number_input("Usia", 1, 120, 30)
    sex = st.selectbox("Jenis Kelamin", ["M", "F"])
    chest_pain = st.selectbox("Tipe Nyeri Dada", ["TA", "ATA", "NAP", "ASY"])
    bp = st.number_input("Tekanan Darah (BP)", 0, 300, 120)
    cholesterol = st.number_input("Kolesterol", 0, 600, 200)
    fbs = st.selectbox("FBS over 120", [0, 1])
    ekg = st.selectbox("Hasil EKG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max HR", 0, 250, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    st_depression_input = st.text_input("ST Depression", "1.0")

    try:
        st_depression = float(st_depression_input.replace(",", "."))
    except ValueError:
        st.error("Masukkan angka valid untuk ST depression.")
        st.stop()

    slope = st.selectbox("Slope of ST", ["Up", "Flat", "Down"])
    vessels = st.selectbox("Jumlah Pembuluh yang Terdeteksi", [0, 1, 2, 3])
    thallium = st.selectbox("Thallium", ["Normal", "Fixed Defect", "Reversable Defect"])
    threshold = st.slider("🎚️ Threshold Risiko", 0.0, 1.0, 0.5, step=0.01)

    if st.button("🔍 Prediksi"):
        # Input data (tanpa encoding manual)
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "Chest pain type": [chest_pain],
            "BP": [bp],
            "Cholesterol": [cholesterol],
            "FBS over 120": [fbs],
            "EKG results": [ekg],
            "Max HR": [max_hr],
            "Exercise angina": [exercise_angina],
            "ST depression": [st_depression],
            "Slope of ST": [slope],
            "Number of vessels fluro": [vessels],
            "Thallium": [thallium]
        })

        # Validasi kolom input
        expected_columns = list(model.feature_names_in_)
        if list(input_df.columns) != expected_columns:
            st.error("❌ Nama/urutan kolom input tidak sesuai dengan yang diharapkan model.")
            st.write("Dibutuhkan:", expected_columns)
            st.write("Diberikan:", list(input_df.columns))
            st.stop()

        try:
            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= threshold)

            st.subheader("📋 Input Pengguna")
            st.dataframe(input_df)

            st.subheader("📈 Hasil Prediksi")
            st.write(f"🔢 **Probabilitas Risiko**: `{prob:.2f}`")
            st.write(f"📉 **Threshold**: `{threshold}`")
            st.write(f"🧠 **Prediksi Kelas**: `{pred}`")

            if pred == 1:
                st.error("⚠️ Pasien berisiko mengalami penyakit jantung.")
            else:
                st.success("✅ Pasien tidak berisiko mengalami penyakit jantung.")

        except Exception as e:
            st.error("❌ Terjadi kesalahan saat prediksi.")
            st.exception(e)
