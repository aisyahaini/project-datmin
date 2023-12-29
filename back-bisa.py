# Panggil library
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Elbow Method
scaler = StandardScaler()
X = scaler.fit_transform(X)

k_values = range(1, 11)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot metode elbow
fig, ax = plt.subplots()
ax.plot(k_values, inertia_values, marker='o')
ax.set_title('Elbow Method based on Inertia (WCSS)')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (WCSS)')
st.pyplot(fig)

# Menampilkan Scatter Plot dengan Warna sesuai Klaster
n_clusters = st.slider("Pilih Jumlah Klaster:", min_value=2, max_value=10, value=3)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
