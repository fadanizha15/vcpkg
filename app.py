import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Memuat model yang telah dilatih
with open('model_uas.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Judul aplikasi
st.title("Prediksi Biaya Asuransi")

# Header tambahan
st.header("Informasi Pengguna")
user_name = st.text_input("Mochammad Fadlan Izha Mayori - 2019230124", "")

# Input dari pengguna
age = st.number_input("Umur", min_value=0, max_value=120, value=27)
sex = st.selectbox("Jenis Kelamin (0: Female, 1: Male)", options=[0, 1])
bmi = st.number_input("Berat Badan (BMI)", min_value=0.0, value=30.0)
children = st.number_input("Jumlah Anak", min_value=0, value=5)
smoker = st.selectbox("Perokok (0: Tidak, 1: Ya)", options=[0, 1])

# Proses input - dijadikan array dan di reshape
X = np.array([age, sex, bmi, children, smoker])
X = X.reshape(1, -1)

# Memprediksi biaya asuransi
if st.button("Prediksi"):
    charger_pred = loaded_model.predict(X)
    st.success(f"Biaya Asuransi Per Bulan: ${charger_pred[0]:.2f}")
