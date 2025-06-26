import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from model import train_model, evaluate_model, load_data

st.set_page_config(page_title="🎓 Rekomendasi Jurusan (XGBoost)", layout="centered")
st.title("🎓 Sistem Rekomendasi Program Studi")

st.markdown("Masukkan data diri kamu untuk mendapatkan saran jurusan yang cocok berdasarkan algoritma **XGBoost**.")

# Simpan riwayat prediksi dalam session
if "history" not in st.session_state:
    st.session_state.history = []

# ✅ Cache model agar tidak dilatih ulang setiap submit
@st.cache_resource
def get_model():
    return train_model()

# Form input
with st.form("form_input"):
    col1, col2 = st.columns(2)
    with col1:
        mat = st.slider("📐 Nilai Matematika", 0, 100, 70)
        ipa = st.slider("🔬 Nilai IPA", 0, 100, 70)
        ips = st.slider("📚 Nilai IPS", 0, 100, 70)
        minat = st.selectbox("🧠 Minat utama kamu", ["Ngoding", "Menggambar", "Menulis", "Berhitung", "Berbicara"])
        karakter = st.selectbox("🎨 Karakter dominan", ["Logis", "Kreatif", "Sabar", "Komunikatif"])
    with col2:
        gaya = st.selectbox("🧩 Gaya belajar", ["Visual", "Auditori", "Kinestetik"])
        tim = st.radio("🤝 Suka kerja tim?", ["Ya", "Tidak"])
        tantangan = st.radio("🔥 Suka tantangan?", ["Ya", "Tidak"])
    
    submit = st.form_submit_button("🔍 Prediksi Jurusan")

# Evaluasi model (sekali tampilkan metrik)
acc = evaluate_model()
st.metric("📈 Akurasi Model", f"{acc*100:.2f}%")

# Submit prediksi
if submit:
    model, encoders = get_model()
    try:
        input_data = np.array([[mat, ipa, ips,
                                encoders['minat'].transform([minat.lower()])[0],
                                encoders['karakter'].transform([karakter.lower()])[0],
                                encoders['gaya_belajar'].transform([gaya.lower()])[0],
                                encoders['suka_kerja_tim'].transform([tim.lower()])[0],
                                encoders['suka_tantangan'].transform([tantangan.lower()])[0]]])
        pred = model.predict(input_data)
        jurusan = encoders['jurusan'].inverse_transform(pred)
        st.success(f"✅ Kamu cocok masuk jurusan: **{jurusan[0]}**")
        st.session_state.history.append(jurusan[0])
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# Riwayat prediksi
if st.session_state.history:
    st.subheader("📊 Riwayat Prediksi Jurusan")
    hist_series = pd.Series(st.session_state.history).value_counts().reset_index()
    hist_series.columns = ["Jurusan", "Jumlah"]
    fig = px.pie(hist_series, names='Jurusan', values='Jumlah',
                 title="Distribusi Jurusan dari Semua Prediksi",
                 hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Belum ada prediksi dilakukan. Isi form di atas untuk mulai.")
