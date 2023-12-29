# Panggil library
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Membuat judul web
st.title("Aplikasi Clustering KMeans")

# File Input
@st.cache_data
def load_dataset(file_path='dataset-clear-fix.csv'):
    # Cek apakah ada input untuk nama file
    if file_path:
        # Jika ada input, gunakan nama file yang diberikan
        dataset = pd.read_csv(file_path)
    else:
        # Jika tidak ada input, gunakan nama file default atau sesuaikan dengan kebutuhan Anda
        dataset = pd.read_csv('data-clear.csv')  # Ganti dengan nama file default yang sesuai

    return dataset

file_path = st.file_uploader("Upload File CSV", type=["csv"])
if file_path is not None:
    dataset = load_dataset(file_path)
else:
    st.warning("Mohon upload file CSV untuk melanjutkan atau gunakan file default.")

# Menampilkan kunci kolom
keys = load_dataset().keys()
st.write("Kunci Kolom:", keys)

# Menampilkan head data
st.write("Data Pertama dalam Dataset:")
st.write(load_dataset().head())

# Konversi ke data Array
X = np.asarray(load_dataset())

# Menampilkan plot kolom sesuai input
column_names = load_dataset().columns.tolist()
selected_column_x = st.selectbox("Pilih Kolom untuk Sumbu X:", column_names)
selected_column_y = st.selectbox("Pilih Kolom untuk Sumbu Y:", column_names)

# Membuat scatter plot berdasarkan pilihan pengguna
fig, ax = plt.subplots()
ax.scatter(X[:, column_names.index(selected_column_x)], X[:, column_names.index(selected_column_y)], label='True Position')
ax.set_xlabel(selected_column_x)
ax.set_ylabel(selected_column_y)
ax.set_title("Peningkatan Jasa Layanan Pengiriman")
st.pyplot(fig)

# Jumlah kluster yang ingin diuji
from sklearn.cluster import KMeans
possible_k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
best_k = 0
best_score = 0

for k in possible_k_values:
    # Inisialisasi model k-means dengan jumlah kluster k
    kmeans = KMeans(n_clusters=k, random_state=20, n_init=1)
    
    # Melakukan klastering
    kmeans.fit(X)
    
    # Menghitung Silhouette Score
    silhouette_avg = silhouette_score(X, kmeans.labels_)

    if(best_score < silhouette_avg):
        best_score = silhouette_avg
        best_k = k
    
    st.write(f"For k={k}, the average silhouette score is: {silhouette_avg}")

st.write(f"Best Cluster : {best_k}")


# Menampilkan Scatter Plot dengan Warna sesuai Klaster
#n_clusters = st.slider("Pilih Jumlah Klaster:", min_value=2, max_value=10, value=3)

kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', s=50)
centroids = ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200,
                       label='Centroids')
ax.set_title('Clustering Results with Cluster Centroids')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend(handles=[centroids])
st.pyplot(fig)
