#panggil library
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

#membuat judul web
st.title(""" Aplikasi Clustering KMeans \n """)

#function untuk ellipse
#Untuk batasan confidence 
#(x => parameter x, y => parameter y, ax => sumbu ellipese, n_std=3.0 => standar deviasi, 
# facecolor='none' => color, **kwargs => argumen, dictionary)
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson) #akar(1+pearson)
    ell_radius_y = np.sqrt(1 - pearson) #akar(1-pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

#membaca data dari file CSV
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = df.values
else:
    #membuat data acak jika tidak ada file CSV yang diunggah
    X = np.random.normal(0, 1, 3000).reshape(-1, 2)  # yang reshape datanya 2 kolom

    # Menampilkan pesan bahwa data acak digunakan
    #st.warning("Tidak ada file CSV yang diunggah. Menggunakan data acak sebagai gantinya.")

    # Menampilkan preview dari data acak
    #st.write("Preview data acak:")
    #st.write(X[:5, :])  # Menampilkan lima baris pertama

#============================================ ELBOW METHOD START ====================================================
# Menambahkan Elbow Method
st.subheader('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=2023)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Menghitung turunan kedua dari penurunan inersia
second_derivative = np.diff(np.diff(inertia_values))

# Mencari indeks titik siku
elbow_index = np.argmax(second_derivative) + 1

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='--')
plt.scatter(elbow_index, inertia_values[elbow_index - 1], c='red', marker='x', label='Elbow Point')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.legend()
st.pyplot()
#======================================== ELBOW METHOD END ========================================================


#klaster untuk label
#cluster_slider = st.slider(
    #min_value=1, max_value=6, value=2, label="Jumlah Cluster :"
#)

#panggil kmeans
kmeans = KMeans(n_clusters=elbow_index, random_state=2023).fit(X)
#label
labels = kmeans.labels_

#mengaktifkan data confidence
seleksi1 = st.selectbox("Visualisasi Batas Confidence?", [False, True])
seleksi2 = st.selectbox("Jumlah Standar Deviasi : ", [1, 2, 3])

#warna yang digunakan
warna = ["red", "blue", "orange", "yellow", "seagreen", "purple"]

#jumlah label untuk ellipse individu
jumlah_label = len(set(labels))

#parameter untuk individu
individu = st.selectbox("Subplot Individu?", [False, True])

#kondisi
if individu:
    fig, ax = plt.subplots(ncols=jumlah_label)
else:
    fig, ax = plt.subplots()
    
#perulangan dari i sampai label, diulang sampai jumlah labelnya
for i, yi in enumerate(set(labels)): 
    if not individu: #jika tidak individu
        a = ax
    else:
        a = ax[i] #jumlah label/cluster
        
    xi = X[labels == yi]
    x_pts = xi[:, 0] #xi dimana semua record nya kolom pertama
    y_pts = xi[:, 1] #xi dimana semua record nya kolom kedua
    a.scatter(x_pts, y_pts, c=warna[yi]) #membuat warna sesuai label
    
    #memilih confidence bond
    if seleksi1:
        #untuk mengaktifkan visualisasi dari batas bond nya
        confidence_ellipse(
            x=x_pts, 
            y=y_pts,
            ax=a,
            edgecolor="black", #warna
            facecolor=warna[yi], #warna sesuai labelnya
            alpha=0.2,
            n_std=seleksi2, #standar deviasi
        )

plt.tight_layout()

# Menambahkan Elbow Method
st.set_option('deprecation.showPyplotGlobalUse', False)


st.subheader(elbow_index)
