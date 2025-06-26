import streamlit as st
from model import train_model, evaluate_model, load_data
import numpy as np

st.set_page_config(page_title="Rekomendasi Program Studi", layout="centered")
st.title("🎓 Rekomendasi Program Studi (Decision Tree)")

st.markdown("Sistem ini akan membantu merekomendasikan jurusan berdasarkan minat, karakter, dan nilai kamu menggunakan algoritma Decision Tree.")

# Input
col1, col2 = st.columns(2)
with col1:
    mat = st.slider("📐 Nilai Matematika", 0, 100, 70)
    ipa = st.slider("🔬 Nilai IPA", 0, 100, 70)
with col2:
    ips = st.slider("📚 Nilai IPS", 0, 100, 70)
    minat = st.selectbox("🧠 Minat utama kamu", ["Ngoding", "Menggambar", "Menulis"])
    karakter = st.selectbox("🎨 Karakter dominan kamu", ["Logis", "Kreatif"])

# Akurasi
acc = evaluate_model()
st.metric("🎯 Akurasi Model Decision Tree", f"{acc*100:.2f}%")

# Prediksi
if st.button("🔍 Rekomendasikan Jurusan"):
    model, le_minat, le_karakter, le_jurusan = train_model()
    X_input = np.array([[mat, ipa, ips,
                         le_minat.transform([minat])[0],
                         le_karakter.transform([karakter])[0]]])
    pred = model.predict(X_input)
    jurusan = le_jurusan.inverse_transform(pred)

    st.success(f"✅ Kamu cocok masuk jurusan: **{jurusan[0]}**")

    penjelasan = {
        "Ngoding": "Kamu punya ketertarikan pada teknologi dan logika.",
        "Menggambar": "Kamu unggul dalam kreativitas dan visualisasi.",
        "Menulis": "Kamu memiliki kemampuan komunikasi dan imajinasi yang baik."
    }
    st.info(penjelasan.get(minat, "Minat kamu sangat unik!") + f" Jurusan **{jurusan[0]}** sesuai dengan karakteristik kamu.")

# Visualisasi dataset
# st.subheader("📊 Distribusi Jurusan di Dataset")
# _, _, _, _, _, df = load_data()

# st.bar_chart(df['jurusan'].value_counts())

# Visualisasi dataset (dengan label asli)
st.subheader("📊 Distribusi Jurusan di Dataset")

# Ambil data + encoder
_, _, _, _, le_jurusan, df = load_data()

# Decode kolom 'jurusan' ke nama asli
df['jurusan_label'] = le_jurusan.inverse_transform(df['jurusan'])

# Tampilkan chart berdasarkan nama jurusan
st.bar_chart(df['jurusan_label'].value_counts())
