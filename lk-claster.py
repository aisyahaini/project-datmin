import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


st.set_page_config(page_title="Segmentasi Customer")

st.title("Segmentasi Customer")

selected = option_menu(
    menu_title=None,
    options=["Home", "Test", "Information"],
    icons=["house", "upload", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    st.header(f" ")

    # Membaca Dataset dari CSV
    df = pd.read_csv("Data Sukun Jual.csv", sep=';')
    # Mengganti nama kolom kdplg menjadi ID Pelanggan
    df.rename(columns={"kdplg": "ID Pelanggan"}, inplace=True)
    st.header("Dataset")
    st.dataframe(df)

    day = "2023-12-01"
    # Mengonversi string tanggal day menjadi objek pandas datetime.
    day = pd.to_datetime(day)
    # Mengatasi kesalahan konversi, dan nilai yang tidak dapat diubah menjadi tanggal akan diisi dengan NaT (Not a Time).
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    # Membuat DataFrame baru yang hanya berisi tahun 2020 - 2023.
    df_filtered = df[df['tanggal'].dt.year.between(2020, 2023)]

    # Recency (waktu terakhir pembelian).
    recency = df_filtered.groupby(["ID Pelanggan"]).agg(
        {"tanggal": lambda x: ((day - x.max()).days)})

    # Frequency (frekuensi pembelian).
    freq = df.drop_duplicates(subset="nota").groupby(
        ["ID Pelanggan"])[["nota"]].count()

    # Monetary (total nilai pembelian).
    df["total"] = df["jumlah"]*df["hgjual"]
    money = df.groupby(["ID Pelanggan"])[["total"]].sum()

    # Menggabungkan hasil Recency, Frequency, dan Monetary ke dalam DataFrame RFM.
    recency.columns = ["Recency"]
    freq.columns = ["Frequency"]
    money.columns = ["Monetary"]
    RFM = pd.concat([recency, freq, money], axis=1)
    st.header("RFM")
    st.write(RFM)

    # Normalisasi Data
    RFM = RFM.fillna(1)  # Mengisi nilai NaN dengan 0 pada DataFrame RFM.

    # menormalkan (scaling) data RFM sehingga memiliki skala yang seragam.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(RFM)

    # Menentukan Jumlah Kluster Optimal.
    inertia = []  # Elbow Method:
    for i in np.arange(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=20)
        kmeans.fit(scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(1, 11), inertia, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    st.header("Elbow Method")
    st.pyplot(fig)

    # Menambahkan elemen input untuk nilai kluster.
    num_clusters = st.number_input(
        "Masukan angka Kluster", min_value=1, max_value=10, value=4)
    # Medeskripsikan kluster sesuai yang di input.
    st.subheader("Deskripsi Kluster")
    if num_clusters == 1:
        st.write(
            f"**Kluster {num_clusters} (Bronze):** Pelanggan dalam kluster ini mungkin terdiri dari pelanggan dengan frekuensi pembelian rendah dan nilai pembelian rendah.")
    elif num_clusters == 2:
        st.write(
            f"**Kluster {num_clusters} (Silver):** Pelanggan dalam kluster ini cenderung memiliki frekuensi pembelian yang tinggi.")
    elif num_clusters == 3:
        st.write(
            f"**Kluster {num_clusters} (Gold):** Kluster ini mungkin mencakup pelanggan dengan nilai pembelian total yang signifikan.")
    elif num_clusters == 4:
        st.write(
            f"**Kluster {num_clusters} (Platinum):** Pelanggan dalam kluster ini memiliki kecenderungan pembelian terakhir yang baru-baru ini.")
    elif num_clusters == 5:
        st.write(
            f"**Kluster {num_clusters} (Diamond):** Pelanggan dalam kluster ini memiliki nilai pembelian yang tinggi dan frekuensi pembelian sedang.")
    elif num_clusters == 6:
        st.write(
            f"**Kluster {num_clusters} (Elite):** Kluster ini mungkin terdiri dari pelanggan dengan frekuensi pembelian rendah dan nilai pembelian rendah.")
    elif num_clusters == 7:
        st.write(
            f"**Kluster {num_clusters} (Premier):** Pelanggan dalam kluster ini cenderung tidak melakukan pembelian baru-baru ini.")
    elif num_clusters == 8:
        st.write(
            f"**Kluster {num_clusters} (Prestige):** Kluster ini mencakup pelanggan dengan frekuensi dan nilai pembelian yang stabil.")
    elif num_clusters == 9:
        st.write(
            f"**Kluster {num_clusters} (Royal):** Pelanggan dalam kluster ini memiliki kecenderungan pembelian yang beragam.")
    elif num_clusters == 10:
        st.write(
            f"**Kluster {num_clusters} (Imperial):** Kluster ini mungkin mencakup pelanggan yang jarang melakukan pembelian, tetapi memiliki nilai pembelian yang tinggi.")

    # Proses clustering setelah menentukan jumlah kluster optimal menggunakan metode siku.
    kmeans = KMeans(n_clusters=num_clusters, random_state=20)
    kmeans.fit(scaled)
    RFM["Kluster"] = (kmeans.labels_ + 1)

    # Melakukan pengelompokkan (grouping) data berdasarkan kolom "Kluster"
    final = RFM.groupby(["Kluster"])[
        ["Recency", "Frequency", "Monetary"]].mean()

    st.header("Average RFM Values by Cluster")
    st.dataframe(final)

    # Visualisasi Hasil Clustering:
    st.header("Clustering Results")
    clustering_results = []
    for i in range(2, 10):
        if i == num_clusters:
            kmeans = KMeans(n_clusters=i, random_state=20)
            kmeans.fit(scaled)
            labels = kmeans.labels_

            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
                3, 2, figsize=(30, 30))  # Membuat 6 petak agar diagram tidak bertabrakan

            # Scatter Plot
            scatter = ax1.scatter(
                scaled[:, 0], scaled[:, 1], c=labels, cmap='viridis')
            ax1.set_title(f"Scatter Plot for {i} Clusters")
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')

            # Menambahkan warna legenda pada scatter plot
            legend = ax1.legend(*scatter.legend_elements(),
                                title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Menambahkan penjelasan warna pada legenda
            legend.set_title("Clusters")
            for text, cluster in zip(legend.get_texts(), range(1, i + 1)):
                text.set_text(f"Cluster {cluster}")

            # Pengertian Warna
            ax2.axis('off')
            ax2.text(0.1, 0.5, 'Pengertian Warna:\nWarna merepresentasikan kluster pada Scatter Plot',
                     fontsize=12, ha='left', va='center')

            # Heatmap Korelasi
            correlation_matrix = RFM.corr()
            sns.heatmap(correlation_matrix, annot=True,
                        cmap='coolwarm', ax=ax3)
            ax3.set_title("Correlation Heatmap")

            # Diagram Batang
            cluster_counts = pd.Series(
                labels).value_counts().sort_index()
            ax4.bar(cluster_counts.index, cluster_counts.values,
                    color=plt.cm.viridis(np.linspace(0, 1, i)))
            ax4.set_title("Cluster Distribution")
            ax4.set_xlabel("Cluster Label")
            ax4.set_ylabel("Count")

            # Box Plot
            sns.boxplot(x='Kluster', y='Recency', data=RFM,
                        palette='viridis', ax=ax5)
            ax5.set_title("Box Plot of Recency by Cluster")

            # Pair Plot
            pair_plot = sns.pairplot(
                RFM, hue='Kluster', palette='viridis', diag_kind='kde', height=3)
            st.write(f"Kluster {i}")
            st.pyplot(pair_plot.fig)

            # Violin Plot
            sns.violinplot(x='Kluster', y='Monetary',
                           data=RFM, palette='viridis', ax=ax6)
            ax6.set_title("Violin Plot of Monetary by Cluster")

            st.pyplot(fig)
            clustering_results.append(
                {"clusters": i, "labels": labels})

    # Interpretasi Hasil Clustering
    def func(row):
        if row["Kluster"] == 1:
            return 'Bronze'
        elif row["Kluster"] == 2:
            return 'Silver'
        elif row["Kluster"] == 3:
            return 'Gold'
        elif row["Kluster"] == 4:
            return 'Platinum'
        elif row["Kluster"] == 5:
            return 'Diamond'
        elif row["Kluster"] == 6:
            return 'Elite'
        elif row["Kluster"] == 7:
            return 'Premier'
        elif row["Kluster"] == 8:
            return 'Prestige'
        elif row["Kluster"] == 9:
            return 'Royal'
        elif row["Kluster"] == 10:
            return 'Imperial'
        elif row["Kluster"] == 11:
            return 'Ultimate'
        else:
            return 'Tidak Diketahui'

    RFM['group'] = RFM.apply(func, axis=1)
    st.header("Hasil Kluster")
    st.write(RFM)

    # Visualisasi Distribusi Kluster
    colors = ["DarkRed", "DarkCyan", "DarkBlue", "Yellow"][:num_clusters]
    result = RFM.group.value_counts()
    st.write(result)

    # Menampilkan Plot Hasil Klaster
    fig_result, ax_result = plt.subplots(figsize=(10, 6))
    result.plot(kind="barh", color=colors)
    ax_result.set_title("Result")
    ax_result.set_xlabel("Count")
    ax_result.set_ylabel("Group")
    st.header("Result")
    st.pyplot(fig_result)

    silhouette_avg = silhouette_score(scaled, RFM["Kluster"])
    st.write("silhouette score: ", silhouette_avg)

    db_index = davies_bouldin_score(scaled, RFM["Kluster"])
    st.write("davies bouldin score: ", db_index)

    ch_index = calinski_harabasz_score(scaled, RFM["Kluster"])
    st.write("calinski harabasz score: ", ch_index)

elif selected == "Test":

    # Upload Data Test
    st.header(f"Upload CSV")
    # Upload file yg bertipe CSV
    data_file = st.file_uploader("Upload CSV", type=["CSV"])

    # Detail File
    if data_file is not None:
        file_details = {"filename": data_file.name,
                        "filetype": data_file.type,
                        "filesize": data_file.size}
        st.write(file_details)

        # Membaca Dataset dari CSV
        df = pd.read_csv(data_file, sep=';')
        # Mengganti nama kolom kdplg menjadi ID Pelanggan
        df.rename(columns={"kdplg": "ID Pelanggan"}, inplace=True)
        st.header("Dataset")
        st.dataframe(df)

        day = "2023-12-01"
        # Mengonversi string tanggal day menjadi objek pandas datetime.
        day = pd.to_datetime(day)
        # Mengatasi kesalahan konversi, dan nilai yang tidak dapat diubah menjadi tanggal akan diisi dengan NaT (Not a Time).
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        # Membuat DataFrame baru yang hanya berisi tahun 2020 - 2023.
        df_filtered = df[df['tanggal'].dt.year.between(2020, 2023)]

        # Recency (waktu terakhir pembelian).
        recency = df_filtered.groupby(["ID Pelanggan"]).agg(
            {"tanggal": lambda x: ((day - x.max()).days)})

        # Frequency (frekuensi pembelian).
        freq = df.drop_duplicates(subset="nota").groupby(
            ["ID Pelanggan"])[["nota"]].count()

        # Monetary (total nilai pembelian).
        df["total"] = df["jumlah"]*df["hgjual"]
        money = df.groupby(["ID Pelanggan"])[["total"]].sum()

        # Menggabungkan hasil Recency, Frequency, dan Monetary ke dalam DataFrame RFM.
        recency.columns = ["Recency"]
        freq.columns = ["Frequency"]
        money.columns = ["Monetary"]
        RFM = pd.concat([recency, freq, money], axis=1)
        st.header("RFM")
        st.write(RFM)

        # Normalisasi Data
        RFM = RFM.fillna(0)  # Mengisi nilai NaN dengan 0 pada DataFrame RFM.

        # menormalkan (scaling) data RFM sehingga memiliki skala yang seragam.
        scaler = StandardScaler()
        scaled = scaler.fit_transform(RFM)

        # Menentukan Jumlah Kluster Optimal.
        inertia = []  # Elbow Method:
        for i in np.arange(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=20)
            kmeans.fit(scaled)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(1, 11), inertia, marker='o')
        ax.set_title('Elbow Method for Optimal k')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        ax.grid(True)
        st.header("Elbow Method")
        st.pyplot(fig)

        # Menambahkan elemen input untuk nilai kluster.
        num_clusters = st.number_input(
            "Masukan angka Kluster", min_value=1, max_value=10, value=4)
        # Medeskripsikan kluster sesuai yang di input.
        st.subheader("Deskripsi Kluster")
        if num_clusters == 1:
            st.write(
                f"**Kluster {num_clusters} (Bronze):** Pelanggan dalam kluster ini mungkin terdiri dari pelanggan dengan frekuensi pembelian rendah dan nilai pembelian rendah.")
        elif num_clusters == 2:
            st.write(
                f"**Kluster {num_clusters} (Silver):** Pelanggan dalam kluster ini cenderung memiliki frekuensi pembelian yang tinggi.")
        elif num_clusters == 3:
            st.write(
                f"**Kluster {num_clusters} (Gold):** Kluster ini mungkin mencakup pelanggan dengan nilai pembelian total yang signifikan.")
        elif num_clusters == 4:
            st.write(
                f"**Kluster {num_clusters} (Platinum):** Pelanggan dalam kluster ini memiliki kecenderungan pembelian terakhir yang baru-baru ini.")
        elif num_clusters == 5:
            st.write(
                f"**Kluster {num_clusters} (Diamond):** Pelanggan dalam kluster ini memiliki nilai pembelian yang tinggi dan frekuensi pembelian sedang.")
        elif num_clusters == 6:
            st.write(
                f"**Kluster {num_clusters} (Elite):** Kluster ini mungkin terdiri dari pelanggan dengan frekuensi pembelian rendah dan nilai pembelian rendah.")
        elif num_clusters == 7:
            st.write(
                f"**Kluster {num_clusters} (Premier):** Pelanggan dalam kluster ini cenderung tidak melakukan pembelian baru-baru ini.")
        elif num_clusters == 8:
            st.write(
                f"**Kluster {num_clusters} (Prestige):** Kluster ini mencakup pelanggan dengan frekuensi dan nilai pembelian yang stabil.")
        elif num_clusters == 9:
            st.write(
                f"**Kluster {num_clusters} (Royal):** Pelanggan dalam kluster ini memiliki kecenderungan pembelian yang beragam.")
        elif num_clusters == 10:
            st.write(
                f"**Kluster {num_clusters} (Imperial):** Kluster ini mungkin mencakup pelanggan yang jarang melakukan pembelian, tetapi memiliki nilai pembelian yang tinggi.")

        # Proses clustering setelah menentukan jumlah kluster optimal menggunakan metode siku.
        kmeans = KMeans(n_clusters=num_clusters, random_state=20)
        kmeans.fit(scaled)
        RFM["Kluster"] = (kmeans.labels_ + 1)

        # Melakukan pengelompokkan (grouping) data berdasarkan kolom "Kluster"
        final = RFM.groupby(["Kluster"])[
            ["Recency", "Frequency", "Monetary"]].mean()

        st.header("Average RFM Values by Cluster")
        st.dataframe(final)

        # Visualisasi Hasil Clustering:
        st.header("Clustering Results")
        clustering_results = []
        for i in range(2, 10):
            if i == num_clusters:
                kmeans = KMeans(n_clusters=i, random_state=20)
                kmeans.fit(scaled)
                labels = kmeans.labels_

                fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
                    3, 2, figsize=(30, 30))  # Membuat 6 petak agar diagram tidak bertabrakan

                # Scatter Plot
                scatter = ax1.scatter(
                    scaled[:, 0], scaled[:, 1], c=labels, cmap='viridis')
                ax1.set_title(f"Scatter Plot for {i} Clusters")
                ax1.set_xlabel('Feature 1')
                ax1.set_ylabel('Feature 2')

                # Menambahkan warna legenda pada scatter plot
                legend = ax1.legend(*scatter.legend_elements(),
                                    title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

                # Menambahkan penjelasan warna pada legenda
                legend.set_title("Clusters")
                for text, cluster in zip(legend.get_texts(), range(1, i + 1)):
                    text.set_text(f"Cluster {cluster}")

                # Pengertian Warna
                ax2.axis('off')
                ax2.text(0.1, 0.5, 'Pengertian Warna:\nWarna merepresentasikan kluster pada Scatter Plot',
                         fontsize=12, ha='left', va='center')

                # Heatmap Korelasi
                correlation_matrix = RFM.corr()
                sns.heatmap(correlation_matrix, annot=True,
                            cmap='coolwarm', ax=ax3)
                ax3.set_title("Correlation Heatmap")

                # Diagram Batang
                cluster_counts = pd.Series(
                    labels).value_counts().sort_index()
                ax4.bar(cluster_counts.index, cluster_counts.values,
                        color=plt.cm.viridis(np.linspace(0, 1, i)))
                ax4.set_title("Cluster Distribution")
                ax4.set_xlabel("Cluster Label")
                ax4.set_ylabel("Count")

                # Box Plot
                sns.boxplot(x='Kluster', y='Recency', data=RFM,
                            palette='viridis', ax=ax5)
                ax5.set_title("Box Plot of Recency by Cluster")

                # Pair Plot
                pair_plot = sns.pairplot(
                    RFM, hue='Kluster', palette='viridis', diag_kind='kde', height=3)
                st.write(f"Kluster {i}")
                st.pyplot(pair_plot.fig)

                # Violin Plot
                sns.violinplot(x='Kluster', y='Monetary',
                               data=RFM, palette='viridis', ax=ax6)
                ax6.set_title("Violin Plot of Monetary by Cluster")

                st.pyplot(fig)
                clustering_results.append(
                    {"clusters": i, "labels": labels})

        # Interpretasi Hasil Clustering
        def func(row):
            if row["Kluster"] == 1:
                return 'Bronze'
            elif row["Kluster"] == 2:
                return 'Silver'
            elif row["Kluster"] == 3:
                return 'Gold'
            elif row["Kluster"] == 4:
                return 'Platinum'
            elif row["Kluster"] == 5:
                return 'Diamond'
            elif row["Kluster"] == 6:
                return 'Elite'
            elif row["Kluster"] == 7:
                return 'Premier'
            elif row["Kluster"] == 8:
                return 'Prestige'
            elif row["Kluster"] == 9:
                return 'Royal'
            elif row["Kluster"] == 10:
                return 'Imperial'
            elif row["Kluster"] == 11:
                return 'Ultimate'
            else:
                return 'Tidak Diketahui'

        RFM['group'] = RFM.apply(func, axis=1)
        st.header("Hasil Kluster")
        st.write(RFM)

        # Visualisasi Distribusi Kluster
        colors = ["DarkRed", "DarkCyan", "DarkBlue", "Yellow"][:num_clusters]
        result = RFM.group.value_counts()
        st.write(result)

        # Menampilkan Plot Hasil Klaster
        fig_result, ax_result = plt.subplots(figsize=(10, 6))
        result.plot(kind="barh", color=colors)
        ax_result.set_title("Result")
        ax_result.set_xlabel("Count")
        ax_result.set_ylabel("Group")
        st.header("Result")
        st.pyplot(fig_result)

elif selected == "Information":
    st.header(f"Halaman 2")
