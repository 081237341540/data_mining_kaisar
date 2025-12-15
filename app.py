import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Clustering + Regression", layout="wide")
st.title("Clustering (KMeans) + Regression (RandomForest)")
st.caption("End to end: load data, clustering, visualisasi, regression, input manual, upload CSV, download hasil.")


# =========================
# UTIL
# =========================
def find_dataset_path():
    candidates = [
        "data/train.csv",
        "train.csv",
        "data/dataset.csv",
        "dataset.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def get_numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(num_cols) == 0:
        return pd.DataFrame()
    out = df[num_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def safe_number(x):
    try:
        return float(x)
    except:
        return np.nan


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Pengaturan")

debug_mode = st.sidebar.checkbox("Tampilkan debug file", value=False)

k_mode = st.sidebar.selectbox("Mode pilih K", ["Manual", "Auto (silhouette)"], index=0)
k_manual = st.sidebar.slider("K (jumlah cluster)", min_value=2, max_value=10, value=3)

test_size = st.sidebar.slider("Test size regression", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.write("Tips: Kalau load lama, pakai K manual 3-5 saja.")


# =========================
# LOAD DATASET
# =========================
path = find_dataset_path()

if debug_mode:
    st.sidebar.write("cwd:", os.getcwd())
    st.sidebar.write("root files:", os.listdir("."))
    if os.path.exists("data"):
        st.sidebar.write("data/ files:", os.listdir("data"))

if path is None:
    st.error("Dataset tidak ditemukan. Pastikan ada `train.csv` di root atau `data/train.csv` di folder data.")
    st.stop()

df_raw = pd.read_csv(path)

st.subheader("Preview Dataset")
st.write(f"Dataset path: `{path}`")
st.write("Shape:", df_raw.shape)
st.dataframe(df_raw.head(20), use_container_width=True)

df_num = get_numeric_clean(df_raw)
if df_num.empty:
    st.error("Dataset tidak memiliki kolom numerik yang bisa dipakai untuk clustering dan regression.")
    st.stop()

st.write("Jumlah kolom numerik:", df_num.shape[1])
st.write("Jumlah baris setelah cleaning (drop NA):", df_num.shape[0])


# =========================
# DEFINE TARGET AND FEATURES
# =========================
num_cols = df_num.columns.tolist()

# rule: target = kolom numerik terakhir
target_col = num_cols[-1]

# fitur = selain target
X_cols = [c for c in num_cols if c != target_col]

if len(X_cols) < 1:
    st.error("Kolom fitur tidak cukup. Minimal harus ada 1 fitur numerik selain target.")
    st.stop()

st.info(f"Target regression dipakai: `{target_col}`. Fitur (X) dipakai: {len(X_cols)} kolom.")


# =========================
# TRAIN CLUSTERING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num[X_cols])

best_k = k_manual

if k_mode == "Auto (silhouette)":
    from sklearn.metrics import silhouette_score

    k_candidates = list(range(2, min(11, max(3, df_num.shape[0] // 5))))
    if len(k_candidates) < 2:
        best_k = 2
    else:
        sil_scores = []
        for k in k_candidates:
            km = KMeans(n_clusters=k, random_state=int(random_state), n_init=10)
            labels_k = km.fit_predict(X_scaled)
            try:
                sil = silhouette_score(X_scaled, labels_k)
            except:
                sil = np.nan
            sil_scores.append(sil)

        # pilih k terbaik
        idx = int(np.nanargmax(sil_scores))
        best_k = k_candidates[idx]

        st.subheader("Silhouette Analysis")
        fig, ax = plt.subplots()
        ax.plot(k_candidates, sil_scores, marker="o")
        ax.set_xlabel("Jumlah Cluster (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score vs K")
        st.pyplot(fig)

st.write("K yang dipakai:", best_k)

kmeans = KMeans(n_clusters=int(best_k), random_state=int(random_state), n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df_work = df_num.copy()
df_work["cluster"] = cluster_labels


# =========================
# VISUALISASI CLUSTER (PCA 2D)
# =========================
st.header("1) Clustering")

st.subheader("Ringkasan Cluster")
c1, c2 = st.columns([1, 2])

with c1:
    counts = pd.Series(cluster_labels).value_counts().sort_index()
    st.dataframe(counts.rename("count").to_frame(), use_container_width=True)

with c2:
    # rata-rata tiap fitur per cluster
    summary = df_work.groupby("cluster")[X_cols].mean(numeric_only=True)
    st.dataframe(summary, use_container_width=True)

st.subheader("Visualisasi Clustering (PCA 2D)")
pca = PCA(n_components=2, random_state=int(random_state))
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="tab10", ax=ax)
ax.set_title("PCA 2D - Cluster")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend(title="Cluster")
st.pyplot(fig)


# =========================
# REGRESSION
# =========================
st.header("2) Regression")

X = df_work[X_cols].copy()
y = df_work[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state)
)

reg = RandomForestRegressor(
    n_estimators=200,
    random_state=int(random_state)
)
reg.fit(X_train, y_train)

y_pred_test = reg.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

m1, m2 = st.columns(2)
with m1:
    st.metric("R2 (test)", f"{r2:.4f}")
with m2:
    st.metric("RMSE (test)", f"{rmse:.4f}")

# prediksi untuk seluruh data
df_work["predicted_target"] = reg.predict(X)

st.subheader("Preview Output (data utama)")
st.dataframe(df_work.head(20), use_container_width=True)

st.download_button(
    "Download output data utama (CSV)",
    data=to_csv_bytes(df_work),
    file_name="output_main_with_cluster_and_prediction.csv",
    mime="text/csv"
)


# =========================
# PREDIKSI DATA BARU
# =========================
st.header("3) Data Baru")

tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])

# ---- TAB 1: INPUT MANUAL ----
with tab1:
    st.subheader("Input Manual")

    with st.form("manual_form"):
        ui_cols = st.columns(3)
        inputs = {}

        for i, col in enumerate(X_cols):
            with ui_cols[i % 3]:
                default_val = float(df_work[col].mean())
                inputs[col] = st.number_input(
                    label=col,
                    value=default_val
                )

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        one = pd.DataFrame([inputs])

        # cluster
        one_scaled = scaler.transform(one)
        c_pred = int(kmeans.predict(one_scaled)[0])

        # regression
        y_pred = float(reg.predict(one)[0])

        a, b = st.columns(2)
        with a:
            st.success(f"Cluster Prediksi: {c_pred}")
        with b:
            st.success(f"Prediksi Target: {y_pred:.4f}")

        st.markdown("Input yang dipakai")
        st.dataframe(one, use_container_width=True)

        # optional: tampilkan posisi titik pada PCA
        try:
            one_pca = pca.transform(one_scaled)
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="tab10", ax=ax2, legend=False)
            ax2.scatter(one_pca[0, 0], one_pca[0, 1], marker="X", s=200)
            ax2.set_title("Posisi Input Manual pada PCA")
            ax2.set_xlabel("PC 1")
            ax2.set_ylabel("PC 2")
            st.pyplot(fig2)
        except:
            pass


# ---- TAB 2: UPLOAD CSV ----
with tab2:
    st.subheader("Upload CSV")
    st.write("File harus punya kolom fitur numerik yang sama seperti fitur (X) di data utama.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload CSV untuk memproses data baru.")
    else:
        df_new_raw = pd.read_csv(uploaded)
        st.write("Preview data baru")
        st.dataframe(df_new_raw.head(20), use_container_width=True)

        missing = [c for c in X_cols if c not in df_new_raw.columns]
        if missing:
            st.error(f"Kolom fitur ini tidak ada di file kamu: {missing}")
        else:
            df_new_feat = df_new_raw[X_cols].copy()
            for c in X_cols:
                df_new_feat[c] = pd.to_numeric(df_new_feat[c], errors="coerce")

            df_new_feat = df_new_feat.replace([np.inf, -np.inf], np.nan).dropna()
            if df_new_feat.empty:
                st.error("Setelah cleaning (konversi numerik dan drop NA), data baru kosong.")
            else:
                X_new_scaled = scaler.transform(df_new_feat)
                cluster_new = kmeans.predict(X_new_scaled)
                pred_new = reg.predict(df_new_feat)

                df_new_out = df_new_raw.loc[df_new_feat.index].copy()
                df_new_out["cluster_pred"] = cluster_new
                df_new_out["predicted_target"] = pred_new

                st.success("Berhasil diproses.")
                st.dataframe(df_new_out.head(50), use_container_width=True)

                st.download_button(
                    "Download output data baru (CSV)",
                    data=to_csv_bytes(df_new_out),
                    file_name="output_new_with_cluster_and_prediction.csv",
                    mime="text/csv"
                )
