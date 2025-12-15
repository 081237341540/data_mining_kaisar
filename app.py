import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Clustering + Regression (Demo)", layout="wide")
st.title("Hasil Processing + Prediksi")
st.caption("Tampilkan ringkasan clustering dan form input untuk prediksi cluster dan target numerik.")


# =========================
# LOAD DATA (auto cari)
# =========================
def find_dataset():
    for p in ["data/train.csv", "train.csv", "data/dataset.csv", "dataset.csv"]:
        if os.path.exists(p):
            return p
    return None

path = find_dataset()
if path is None:
    st.error("Dataset tidak ditemukan. Pastikan ada train.csv di root atau data/train.csv.")
    st.stop()

df_raw = pd.read_csv(path)

# ambil numerik saja, drop NA
df_num = df_raw.select_dtypes(include=["int64", "float64"]).replace([np.inf, -np.inf], np.nan).dropna()
if df_num.shape[1] < 2:
    st.error("Kolom numerik kurang dari 2. Clustering + regression butuh minimal 2 kolom numerik.")
    st.stop()

num_cols = df_num.columns.tolist()
target_col = num_cols[-1]
X_cols = [c for c in num_cols if c != target_col]


# =========================
# SIDEBAR SETTING (minimal)
# =========================
st.sidebar.header("Pengaturan")
k = st.sidebar.slider("Jumlah cluster (K)", 2, 10, 3)
test_size = st.sidebar.slider("Test size regression", 0.1, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input("Random state", value=42, step=1)


# =========================
# TRAIN (di belakang layar)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num[X_cols])

kmeans = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
labels = kmeans.fit_predict(X_scaled)

# regression
X = df_num[X_cols].copy()
y = df_num[target_col].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(seed))

reg = RandomForestRegressor(n_estimators=200, random_state=int(seed))
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))


# =========================
# UI: HASIL PROCESSING
# =========================
st.header("1) Hasil Processing (Ringkasan)")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Jumlah data (setelah cleaning)", df_num.shape[0])
with c2:
    st.metric("Jumlah fitur (X)", len(X_cols))
with c3:
    st.metric("Target regression", target_col)

st.subheader("Ringkasan Cluster")
colA, colB = st.columns([1, 2])

with colA:
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    st.dataframe(cluster_counts.rename("count").to_frame(), use_container_width=True)

with colB:
    df_tmp = df_num.copy()
    df_tmp["cluster"] = labels
    profile = df_tmp.groupby("cluster")[X_cols].mean(numeric_only=True)
    st.dataframe(profile, use_container_width=True)

st.subheader("Visualisasi Clustering (PCA 2D)")
pca = PCA(n_components=2, random_state=int(seed))
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", ax=ax)
ax.set_title("PCA 2D - Cluster")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend(title="Cluster")
st.pyplot(fig)

st.subheader("Kinerja Regression (sekilas)")
m1, m2 = st.columns(2)
with m1:
    st.metric("R2 (test)", f"{r2:.4f}")
with m2:
    st.metric("RMSE (test)", f"{rmse:.4f}")


# =========================
# UI: INPUT PREDIKSI
# =========================
st.header("2) Input untuk Prediksi")

st.write("Isi nilai fitur di bawah, lalu klik Prediksi. Output: cluster + prediksi target.")

with st.form("predict_form"):
    ui_cols = st.columns(3)
    inputs = {}

    for i, col in enumerate(X_cols):
        with ui_cols[i % 3]:
            default_val = float(df_num[col].mean())
            inputs[col] = st.number_input(col, value=default_val)

    submit = st.form_submit_button("Prediksi")

if submit:
    one = pd.DataFrame([inputs])

    # cluster pred
    one_scaled = scaler.transform(one)
    cluster_pred = int(kmeans.predict(one_scaled)[0])

    # regression pred
    pred_target = float(reg.predict(one)[0])

    a, b = st.columns(2)
    with a:
        st.success(f"Cluster Prediksi: {cluster_pred}")
    with b:
        st.success(f"Prediksi {target_col}: {pred_target:.4f}")

    # tampilkan posisi input di PCA
    try:
        one_pca = pca.transform(one_scaled)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", ax=ax2, legend=False)
        ax2.scatter(one_pca[0, 0], one_pca[0, 1], marker="X", s=220)
        ax2.set_title("Posisi Input pada PCA")
        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")
        st.pyplot(fig2)
    except:
        pass
