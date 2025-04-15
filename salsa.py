import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sistem Rekrutmen K-Means", layout="centered")

st.title("📊 Sistem Rekrutmen Berbasis K-Means Clustering")
st.markdown("<h6 style='text-align: center;'>by muslim safiq</h6>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload file CSV pelamar", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Pelamar")
    st.dataframe(df)

    fitur_numerik = ['ipk', 'pengalaman_kerja', 'nilai_skill', 'nilai_wawancara']

    if all(col in df.columns for col in fitur_numerik):
        # Normalisasi
        scaler = StandardScaler()
        X = scaler.fit_transform(df[fitur_numerik])

        # Pilih jumlah klaster
        k = st.slider("Pilih jumlah klaster", 2, 3, 5)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['klaster'] = kmeans.fit_predict(X)

        st.subheader("🔍 Hasil Klasterisasi")
        st.dataframe(df[['nama'] + fitur_numerik + ['klaster']])

        # Visualisasi
        st.subheader("📈 Visualisasi Klaster (2 Fitur)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['ipk'], y=df['pengalaman_kerja'], hue=df['klaster'], palette="Set2", ax=ax)
        plt.xlabel("IPK")
        plt.ylabel("Pengalaman Kerja")
        st.pyplot(fig)

        # Kandidat terbaik (misal klaster dengan jumlah terbanyak)
        best_cluster = df['klaster'].value_counts().idxmax()
        st.subheader(f"🏅 Kandidat Potensial (Klaster {best_cluster})")
        st.dataframe(df[df['klaster'] == best_cluster][['nama', 'klaster']])
    else:
        st.warning("Kolom yang dibutuhkan: ipk, pengalaman_kerja, nilai_skill, nilai_wawancara")
