#panggil library
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#membuat judul web
st.title(""" Aplikasi Clustering KMeans \n """)

#====================================== FIle Input ======================================================
def load_dataset(file_path='dataset-clear-fixs.csv'):
    # Cek apakah ada input untuk nama file
    if file_path:
        # Jika ada input, gunakan nama file yang diberikan
        dataset = pd.read_csv(file_path)
    else:
        # Jika tidak ada input, gunakan nama file default atau sesuaikan dengan kebutuhan Anda
        dataset = pd.read_csv('data-clear.csv')  # Ganti dengan nama file default yang sesuai

    return dataset

# Meminta input dari pengguna untuk nama file (dapat dikosongkan)
user_input = input("Masukkan nama file CSV atau biarkan kosong untuk menggunakan file default: ")

# Memanggil fungsi untuk memuat dataset
dataset = load_dataset(user_input)

# Menampilkan kunci kolom
keys = dataset.keys()
print(keys)
#==================================== File Input End ============================================

#==================================== Menampilkan head data =====================================
dataku = pd.DataFrame(dataset)
dataku.head()
#==================================== Menampilkan head data End =====================================

#Konversi ke data Array
X = np.asarray(dataset)
print(X)

#==================================== Menampilkan plot kolom sesuai input ================================
# Misalnya, X memiliki tiga kolom ("Status", "Order Account", "Payment Method")
# Membuat daftar nama kolom
column_names = ["Status", "Order Account", "Payment Method"]

# Meminta input dari pengguna untuk memilih kolom
selected_column_x = int(input("Masukkan indeks kolom untuk sumbu x: "))
selected_column_y = int(input("Masukkan indeks kolom untuk sumbu y: "))

# Membuat scatter plot berdasarkan pilihan pengguna
plt.scatter(X[:, selected_column_x], X[:, selected_column_y], label='True Position')
plt.xlabel(column_names[selected_column_x])
plt.ylabel(column_names[selected_column_y])
plt.title("Peningkatan Jasa Layanan Pengiriman")
plt.show()
#==================================== Menampilkan plot kolom sesuai input End ================================

#==================================== Elbow Method (sementara) ===============================================
# Misalnya, X memiliki beberapa kolom numerik
# Gantilah ini dengan data aktual Anda
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Inisialisasi rentang nilai k yang akan diuji
k_values = range(1, 11)  # Coba k dari 1 hingga 10

# Inisialisasi list untuk menyimpan nilai inertia
inertia_values = []

# Uji setiap nilai k
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X)

    # Simpan nilai inertia (WCSS)
    inertia_values.append(kmeans.inertia_)

# Plot metode elbow berdasarkan inertia (WCSS)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method based on Inertia (WCSS)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')

# Temukan titik "elbow" (poin di mana penurunan inertia melambat)
diff = np.diff(inertia_values)
elbow_point = k_values[np.argmax(diff) + 1]  # +1 karena diff mengurangi satu elemen

print(f"Optimal number of clusters (Elbow Method): {elbow_point}")

plt.show()
#==================================== Elbow Method (sementara) End ===============================================

#Menampilkan nilai Centroid yang digenerate oleh algoritma
print(kmeans.cluster_centers_)

#==================================== Scatter Atribut sesuai pilihan ============================
# Misalnya, X memiliki tiga kolom ("Status", "Order Account", "Payment Method")

# Membuat daftar nama kolom
column_names = ["Status", "Order Account", "Payment Method"]

# Meminta input dari pengguna untuk memilih kolom
selected_column_x = int(input("Masukkan indeks kolom untuk sumbu x: "))
selected_column_y = int(input("Masukkan indeks kolom untuk sumbu y: "))

# Membuat scatter plot berdasarkan pilihan pengguna
plt.scatter(X[:, selected_column_x], X[:, selected_column_y], c=kmeans.labels_, cmap='rainbow')
plt.xlabel(column_names[selected_column_x])
plt.ylabel(column_names[selected_column_y])
plt.title("Peningkatan Jasa Layanan Pengiriman")
plt.show()
#==================================== Scatter Atribut sesuai pilihan End ============================

#==================================== Menampilkan hasil sesuai jumlah data ====================================
# Membuat data dummy sebagai contoh
X = np.array([
    [1, 2],
    [5, 8],
    [1.5, 1.8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])

# Menentukan jumlah cluster yang optimal (dapat disesuaikan dengan analisis sebelumnya)
n_clusters = 3

# Melakukan proses clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Menghitung jumlah data pada setiap klaster
cluster_counts = np.bincount(cluster_labels)

# Menambahkan informasi klaster dan jumlah data ke data
X_with_labels = np.column_stack((X, cluster_labels))

# Menampilkan hasil klastering dengan informasi jumlah data
plt.scatter(X_with_labels[:, 0], X_with_labels[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')

# Menambahkan label untuk setiap data point
for i in range(len(X)):
    plt.text(X_with_labels[i, 0], X_with_labels[i, 1], f"{int(X_with_labels[i, 2])}\n({cluster_counts[int(X_with_labels[i, 2])]} data)", fontsize=10, ha='center', va='center')

plt.title('Clustering Results with Cluster Centroids and Data Count')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
#==================================== Menampilkan hasil sesuai jumlah data End ====================================