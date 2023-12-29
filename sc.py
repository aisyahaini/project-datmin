import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import make_blobs

st.set_option('deprecation.showPyplotGlobalUse', False)

###
debug = st.sidebar.checkbox('Show Debug', value=False)

ntn = st.sidebar.checkbox('Mover Centroids com média NaN para (0,0)', value=True)


def distancia(p1, p2):
    """
    Método que recebe dois pontos, e retorna a distância euclidiana entre eles.
    """
    return np.sqrt(np.sum((p1 - p2)**2))


def do_kmeans_clustering(k, X, random_state):

    global debug

    np.random.seed(random_state)

    # centroids = np.random.rand(k, 2)
    centroids = np.random.randint(-10, 10, (k, 2))
    centers = centroids

    st.markdown('Centroids Iniciais:')
    st.write(centers)

    labels = np.zeros((len(X)))
    distancias = np.zeros((len(X), k))

    passo = 1

    while True:
        if debug:
            print(f'\n-- Passo {passo}')

        old_centers = centers

        for i in range(k):  # de 0 a 2
            for j in range(len(X)):  # de 0 a 300
                # print(f'\n[Passo {passo}] Calculando distancia do ponto {X[j]} para centroid {centroids[i]}')
                # print(distancia(X[j], centroids[i]))
                distancias[j, i] = distancia(X[j], centers[i])

        labels = np.argmin(distancias, axis=1)  # Define as labels como o indice de menor valor no array de distâncias

        centers = np.array([X[labels == i].mean(0) for i in range(k)])  # Calcula a média dos pontos de cada cluster

        if ntn:
            centers = np.nan_to_num(centers)  # Se um cluster estiver zerado, transforma np.NaN em 0

        if debug:
            st.markdown(f'DEBUG | Passo {passo}')
            sns.scatterplot(x='x', y='y', data=df, hue=labels, palette='rainbow', legend=False)
            plt.scatter(old_centers[:, 0], old_centers[:, 1], color='black', s=100)
            plt.title(f'DEBUG = Passo {passo}')
            st.pyplot()
            st.markdown('Distâncias: ')
            st.write(distancias)
            st.markdown('Centroids Anteriores: ')
            st.write(old_centers)
            st.markdown('Novos Centroids: ')
            st.write(centers)
            st.markdown('---')

        if np.array_equal(centers, old_centers):
            if debug:
                print('Centers', centers)
                print('Old Centers', old_centers)
            break

        if debug:
            # print(distancias)
            print(labels)
            print(centers)
            print('-' * 10)
        passo = passo + 1

        # Break adicionado para fins de teste
        if passo > 30:
            st.markdown('Foi adicionado um limite de 30 passos para evitar loops infinitos.')
            break

    centroids = np.array(centroids)
    return centers, labels


def plot(X, centroids, labels, random_state):
    sns.scatterplot(x='x', y='y', data=X, hue=labels, palette='rainbow', legend=False)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', s=100)
    plt.title(f'Final, com {len(centroids)} clusters e RandomState {random_state} e {len(df)} pontos.')
    st.pyplot()


###


st.markdown('# Implementação Manual K-Means')

"""
Ferramenta criada na aula de `Aprendizado Não Supervisionado - Agrupamento de Dados`.

#### Prof. Thomas da Silva

#### Alunos:
- Lucas Knaak
- William Becher
"""

plt.rcParams['figure.figsize'] = (14, 8)
sns.set(context='talk')
centers = 3
n_features = 2

n_samples = st.sidebar.number_input('n_samples', min_value=10, max_value=1000, value=300, step=50)

# if st.checkbox('Ver algoritmo'):
#     """
# ```python
# def do_kmeans_clustering(k, X, random_state):
#     # Código Pronto Vai Aqui
#     return centers, labels
# ```
#     """

random_state = 100
X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=random_state)
df = pd.DataFrame(X, columns=['x', 'y'])

# sns.scatterplot(x='x', y='y', data=df, palette='rainbow', legend=False)
# plt.title('Dados Iniciais Aleatórios')
# st.pyplot()

k = st.sidebar.number_input('k', min_value=1, max_value=100, value=3)
random_state = st.sidebar.number_input('Random State', min_value=1, max_value=10000, value=30)

f"""
Parâmetros utilizados:
- k = {k}
- random_state = {random_state}
"""

centroids, labels = do_kmeans_clustering(k, X, random_state)
plot(df, centroids, labels, random_state)