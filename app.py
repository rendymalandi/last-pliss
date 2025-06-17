# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_heart_disease_pipeline.pkl")

# Sidebar navigasi
st.sidebar.title("Navigasi")
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
    st.markdown("- Model: `xgboost_heart_disease_pipeline.pkl`")
    try:
        url = "https://raw.githubusercontent.com/rendymalandi/last-pliss/main/Heart_Disease_Prediction.csv"
        df = pd.read_csv(url)
        from sklearn.metrics import classification_report
        X = df.drop("Heart Disease", axis=1)
        y = df["Heart Disease"]
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        st.json(report)
    except Exception as e:
        st.warning(f"Error saat evaluasi model: {e}")

# Halaman Prediksi
elif page == "Prediksi":
    st.title("🩺 Prediksi Penyakit Jantung")

    # Input sesuai nama kolom training
    age = st.number_input("Usia", 1, 120, 30)
    sex = st.selectbox("Jenis Kelamin", ["M", "F"])
    chest_pain = st.selectbox("Tipe Nyeri Dada", ["TA", "ATA", "NAP", "ASY"])
    bp = st.number_input("Tekanan Darah (BP)", 0, 300, 120)
    cholesterol = st.number_input("Kolesterol", 0, 600, 200)
    fbs = st.selectbox("FBS over 120", [0, 1])
    ekg = st.selectbox("EKG results", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max HR", 0, 250, 150)
    exercise_angina = st.selectbox("Exercise angina", ["Y", "N"])
    st_depression = st.text_input("ST depression", value="1.0")
    slope = st.selectbox("Slope of ST", ["Up", "Flat", "Down"])
    vessels = st.selectbox("Number of vessels fluro", [0, 1, 2, 3])
    thallium = st.selectbox("Thallium", ["Normal", "Fixed Defect", "Reversable Defect"])

    threshold = st.slider("🎚️ Threshold Risiko", 0.0, 1.0, 0.5, step=0.01)

    if st.button("Prediksi"):
        try:
            st_depression_float = float(str(st_depression).replace(",", "."))
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
                "ST depression": [st_depression_float],
                "Slope of ST": [slope],
                "Number of vessels fluro": [vessels],
                "Thallium": [thallium]
            })

            # Pastikan kategori terdeteksi
            categorical = [
                "Sex", "Chest pain type", "EKG results", "Exercise angina",
                "Slope of ST", "Thallium"
            ]
            for col in categorical:
                input_df[col] = input_df[col].astype("category")

            prob = model.predict_proba(input_df)[0][1]  # Probabilitas kelas 1 (berisiko)
            pred = int(prob >= threshold)

            # Debug info tambahan
            st.write("\n🧾 Input dikirim ke model:")
            st.dataframe(input_df)
            st.write(f"🔢 **Probabilitas Risiko (kelas 1)**: `{prob:.2f}`")
            st.write(f"📈 **Threshold saat ini**: `{threshold}`")
            st.write(f"📊 **Prediksi Kelas**: `{pred}`")

            if pred == 1:
                st.error("⚠️ Pasien berisiko mengalami penyakit jantung.")
            else:
                st.success("✅ Pasien tidak berisiko mengalami penyakit jantung.")

        except Exception as e:
            st.error("❌ Terjadi kesalahan saat prediksi. Mohon periksa kembali input Anda.")
            st.exception(e)
