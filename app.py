import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from model import train_model, evaluate_model, load_data  # pastikan file model.py sudah siap

# ====== Konfigurasi Halaman ======
st.set_page_config(page_title="🎓 Rekomendasi Jurusan (XGBoost)", layout="centered")
st.title("🎓 Sistem Rekomendasi Program Studi")

st.markdown("Masukkan data diri kamu untuk mendapatkan saran jurusan yang cocok berdasarkan algoritma **XGBoost**.")

# ====== Bahasa ======
language = st.selectbox("🌐 Pilih Bahasa", ["Bahasa Indonesia", "English"])

# ====== Riwayat Prediksi (Session State) ======
if "history" not in st.session_state:
    st.session_state.history = []

# ====== Cache model agar tidak dilatih ulang ======
@st.cache_resource
def get_model():
    return train_model()

# ====== Tambahan Input Fitur ======
st.markdown("### 🧩 Tambahan Informasi Personal")
sekolah = st.selectbox("🏫 Jenis Sekolah", ["SMA IPA", "SMA IPS", "SMK TKJ", "SMK Akuntansi", "MA", "Homeschooling"])
lokasi = st.selectbox("🌍 Lokasi studi yang diinginkan", ["Jawa Barat", "DKI Jakarta", "Yogyakarta", "Luar Negeri"])
tujuan = st.selectbox("🎯 Saat kuliah nanti, kamu ingin lebih fokus ke:", [
    "Ilmu pengetahuan dan riset",
    "Bekerja langsung di masyarakat",
    "Mengekspresikan kreativitas",


    "Membangun karier secepatnya"
])
tipe_pemecah = st.selectbox("🧠 Saat hadapi masalah, kamu cenderung:", [
    "Menganalisis logika", 
    "Diskusi bareng teman", 
    "Cari solusi kreatif", 
    "Bikin rencana terstruktur"
])

# ====== Form Input ======
with st.form("form_input"):
    col1, col2 = st.columns(2)
    with col1:
        mat = st.slider("📐 Nilai Matematika", 0, 100, 70)
        ipa = st.slider("🔬 Nilai IPA", 0, 100, 70)
        ips = st.slider("📚 Nilai IPS", 0, 100, 70)
        bindo = st.slider("📝 Nilai Bahasa Indonesia", 0, 100, 70)
        bing = st.slider("🌍 Nilai Bahasa Inggris", 0, 100, 70)
        minat = st.selectbox("🧠 Minat utama kamu", ["Ngoding", "Menggambar", "Menulis", "Berhitung", "Berbicara"])
        karakter = st.selectbox("🎨 Karakter dominan", ["Logis", "Kreatif", "Sabar", "Komunikatif"])
    with col2:
        gaya = st.selectbox("🧩 Gaya belajar", ["Visual", "Auditori", "Kinestetik"])
        kerja = st.radio("🤝 Suka kerja tim?", ["Ya", "Tidak"])
        tantangan = st.radio("🔥 Suka tantangan?", ["Ya", "Tidak"])

    submit = st.form_submit_button("🔍 Prediksi Jurusan")

# ====== Evaluasi model (hanya sekali tampilkan) ======
acc = evaluate_model()
st.metric("📈 Akurasi Model", f"{acc*100:.2f}%")

# ====== Submit Prediksi ======
if submit:
    model, encoders = get_model()
    try:
        input_data = np.array([[mat, ipa, ips, bindo, bing,
                                encoders['minat'].transform([minat.lower()])[0],
                                encoders['karakter'].transform([karakter.lower()])[0],
                                encoders['gaya_belajar'].transform([gaya.lower()])[0],
                                encoders['suka_kerja_tim'].transform([kerja.lower()])[0],
                                encoders['suka_tantangan'].transform([tantangan.lower()])[0],
                                encoders['sekolah'].transform([sekolah.lower()])[0],
                                encoders['lokasi'].transform([lokasi.lower()])[0],
                                encoders['tipe_pemecah'].transform([tipe_pemecah.lower()])[0],
                                encoders['tujuan'].transform([tujuan.lower()])[0]
                                
                               ]])

        probs = model.predict_proba(input_data)[0]
        top3_idx = probs.argsort()[-3:][::-1]
        top3_jurusan = encoders['jurusan'].inverse_transform(top3_idx)
        top3_probs = probs[top3_idx]

        st.success(f"✅ Prediksi Jurusan Teratas:")
        for jur, p in zip(top3_jurusan, top3_probs):
            st.write(f"🎓 {jur} — {p*100:.1f}%")

        st.session_state.history.append(top3_jurusan[0])

        # Tampilkan deskripsi jurusan (jika tersedia)
        st.markdown("---")
        if top3_jurusan[0] == "Teknik Informatika":
            st.info("📘 *Teknik Informatika mempelajari tentang pemrograman, sistem komputer, dan pengembangan aplikasi.*")
        elif top3_jurusan[0] == "Desain Komunikasi Visual":
            st.info("🎨 *DKV menggabungkan seni dan komunikasi visual untuk media digital atau cetak.*")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ====== Riwayat Prediksi ======
if st.session_state.history:
    st.subheader("📊 Riwayat Prediksi Jurusan")
    hist_series = pd.Series(st.session_state.history).value_counts().reset_index()
    hist_series.columns = ["Jurusan", "Jumlah"]
    fig = px.pie(hist_series, names='Jurusan', values='Jumlah',
                 title="Distribusi Jurusan dari Semua Prediksi", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Belum ada prediksi dilakukan. Isi form di atas untuk mulai.")
