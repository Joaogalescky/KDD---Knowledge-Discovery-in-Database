"""
SVM - Breast Cancer Wisconsin Dataset
"""


# 1. IMPORTAÇÃO DAS BIBLIOTECAS


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report
)

from sklearn.decomposition import PCA


# 2. CARREGAMENTO DO DATASET


cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

print("=" * 60)
print("Dataset carregado com sucesso")
print(f"Amostras : {x.shape[0]}")
print(f"Features : {x.shape[1]}")
print("=" * 60)


# 3. PREPARAÇÃO DOS DADOS


# utilizar apenas 50% do dataset (opcional)

df = pd.DataFrame(x, columns=cancer.feature_names)
df["target"] = y

df = df.sample(frac=0.5, random_state=42)

x = df.drop("target", axis=1)
y = df["target"]


# divisão treino/teste

X_train, x_teste, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.30,
    random_state=109,
    stratify=y
)  # 30% teste | 70% treinamento


# normalização

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
x_teste = scaler.transform(x_teste)


# 4. TREINAMENTO DO MODELO


modelo = SVC(
    kernel="linear",
    random_state=109
)

modelo.fit(X_train, y_train)

print("\nModelo treinado com sucesso.\n")


# 5. AVALIAÇÃO


predicoes = modelo.predict(x_teste)

print("Acurácia")
print(
    accuracy_score(
        y_teste,
        predicoes
    )
)
print("\n")

print("Precisão")
print(
    precision_score(
        y_teste,
        predicoes
    )
)
print("\n")

print("Matriz de Confusão")
print(
    confusion_matrix(
        y_teste,
        predicoes
    )
)
print("\n")

print("Relatório")
print(
    classification_report(
        y_teste,
        predicoes,
        target_names=cancer.target_names
    )
)


# 6. PCA (SOMENTE PARA VISUALIZAÇÃO)


pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)

modelo_pca = SVC(
    kernel="linear",
    random_state=109
)

modelo_pca.fit(
    X_train_pca,
    y_train
)


# 7. FRONTEIRA DE DECISÃO


x_min = X_train_pca[:, 0].min() - 1
x_max = X_train_pca[:, 0].max() + 1

y_min = X_train_pca[:, 1].min() - 1
y_max = X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.10),
    np.arange(y_min, y_max, 0.10)
)

Z = modelo_pca.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.35,
    cmap="coolwarm"
)

sns.scatterplot(
    x=X_train_pca[:, 0],
    y=X_train_pca[:, 1],
    hue=y_train,
    palette="viridis",
    edgecolor="black",
    s=70
)

plt.scatter(
    modelo_pca.support_vectors_[:, 0],
    modelo_pca.support_vectors_[:, 1],
    facecolors="none",
    edgecolors="red",
    s=180,
    linewidths=2,
    label="Vetores de suporte"
)

plt.title("SVM - Fronteira de decisão (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")

plt.grid(True)

plt.legend()

plt.show()
