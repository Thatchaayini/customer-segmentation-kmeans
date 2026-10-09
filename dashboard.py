import plotly.express as px
import streamlit as st

from evaluation import evaluate_clustering
from preprocess import preprocess
from segmentation import segment_customers, cluster_summary
from visualization import reduce_pca, reduce_tsne

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")

FEATURES = ["Age", "AnnualIncome", "PurchaseFrequency", "AvgOrderValue"]


@st.cache_data
def load_data():
    df, X_scaled, _ = preprocess()
    return df, X_scaled


@st.cache_data
def get_tsne(X_scaled):
    return reduce_tsne(X_scaled)


df, X_scaled = load_data()

# Sidebar controls
st.sidebar.header("Controls")
k = st.sidebar.slider("Number of clusters (K)", 2, 8, 3)
method = st.sidebar.radio("Projection", ["PCA", "t-SNE"])

result, model, used_k = segment_customers(df, X_scaled, k)
metrics = evaluate_clustering(X_scaled, result["Cluster"].to_numpy())

# Metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clusters used", used_k)
c2.metric("Silhouette (higher is better)", metrics["silhouette"])
c3.metric("Calinski-Harabasz (higher is better)", metrics["calinski_harabasz"])
c4.metric("Davies-Bouldin (lower is better)", metrics["davies_bouldin"])

# 2D projection
if method == "PCA":
    coords, variance = reduce_pca(X_scaled)
    st.caption(f"PCA explains {variance.sum():.1%} of the variance")
else:
    coords = get_tsne(X_scaled)

plot_df = result.copy()
plot_df["x"] = coords[:, 0]
plot_df["y"] = coords[:, 1]
plot_df["Cluster"] = plot_df["Cluster"].astype(str)

clusters = sorted(plot_df["Cluster"].unique())
selected = st.sidebar.multiselect("Show clusters", clusters, default=clusters)
view = plot_df[plot_df["Cluster"].isin(selected)]

left, right = st.columns(2)

with left:
    st.subheader(f"Customer segments ({method})")
    fig = px.scatter(
        view, x="x", y="y", color="Cluster",
        hover_data=["CustomerID"] + FEATURES,
        category_orders={"Cluster": clusters},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    feature = st.selectbox("Compare a feature across clusters", FEATURES)
    st.subheader(f"{feature} by cluster")
    box = px.box(
        view, x="Cluster", y=feature, color="Cluster",
        category_orders={"Cluster": clusters},
    )
    st.plotly_chart(box, use_container_width=True)

st.subheader("Cluster profiles (average values)")
st.dataframe(cluster_summary(result), use_container_width=True)

st.download_button(
    "Download segmented data (CSV)",
    result.to_csv(index=False),
    file_name="customers_segmented.csv",
    mime="text/csv",
)