import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Clustering + Regression", layout="wide")
st.title("Clustering + Regression (End-to-End)")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("train.csv")
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df = df[num_cols].dropna()

# =========================
# CLUSTERING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# =========================
# VISUALISASI
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.subheader("Visualisasi Clustering (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="tab10", ax=ax)
st.pyplot(fig)

# =========================
# REGRESSION
# =========================
target = num_cols[-1]
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

df["prediction"] = reg.predict(X)

st.subheader("Preview Data dengan Cluster & Prediksi")
st.dataframe(df.head(20))
