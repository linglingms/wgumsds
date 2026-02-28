import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px


st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.title("Interactive Clustering Dashboard (D603 Task 2)")

# Try loading cleaned CSV first, fall back to original Excel if needed
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('d603task2_cleaned_data.csv')
        source = 'd603task2_cleaned_data.csv'
    except Exception:
        try:
            df = pd.read_excel('medical_clean_d603.xlsx')
            source = 'medical_clean_d603.xlsx'
        except Exception:
            st.error('Could not find data files: d603task2_cleaned_data.csv or medical_clean_d603.xlsx')
            return None, None
    return df, source


data, source = load_data()
if data is None:
    st.stop()

st.markdown(f"**Loaded data:** {source} — shape {data.shape}")

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error('No numeric columns found in the dataset.')
    st.stop()

with st.sidebar:
    st.header('Controls')
    features = st.multiselect('Select features for clustering', options=numeric_cols, default=numeric_cols)
    scale_option = st.checkbox('Scale features (StandardScaler)', value=False)
    algo = st.radio('Clustering algorithm', ['KMeans', 'Agglomerative'], index=0)
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    show_elbow = st.checkbox('Show Elbow & Silhouette (KMeans only)', value=True)
    random_state = st.number_input('Random seed', value=42)
    run = st.button('Run clustering')

if len(features) < 2:
    st.warning('Pick at least 2 features for PCA visualization.')

def maybe_scale(X, do_scale):
    if not do_scale:
        return X, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def compute_elbow_silhouette(X):
    inertias = []
    silhs = []
    ks = list(range(1, 11))
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        if k >= 2:
            labels = kmeans.labels_
            silhs.append(silhouette_score(X, labels))
        else:
            silhs.append(None)
    return ks, inertias, silhs

if run:
    X = data[features].copy()

    # Heuristic: if data columns appear already standardized (mean ~0, std ~1), default to no scaling
    auto_scaled = all(abs(X[col].mean()) < 1e-1 and abs(X[col].std() - 1) < 0.5 for col in X.columns)
    if auto_scaled and not scale_option:
        st.info('Numeric columns look pre-scaled (mean≈0, std≈1). Scaling skipped.')

    X_for_model, scaler = maybe_scale(X, scale_option and not auto_scaled)

    if show_elbow and algo == 'KMeans':
        ks, inertias, silhs = compute_elbow_silhouette(X_for_model)
        fig_elbow = px.line(x=ks, y=inertias, markers=True, labels={'x':'k','y':'Inertia'}, title='Elbow Curve (Inertia)')
        fig_silh = px.line(x=ks[1:], y=silhs[1:], markers=True, labels={'x':'k','y':'Silhouette'}, title='Silhouette Score (k>=2)')
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.plotly_chart(fig_silh, use_container_width=True)

    # Fit clustering
    if algo == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X_for_model)
        centers = model.cluster_centers_
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_for_model)
        centers = None

    data_results = data.copy()
    data_results['cluster'] = labels.astype(str)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    emb = pca.fit_transform(X_for_model)
    fig = px.scatter(
        x=emb[:,0], y=emb[:,1], color=data_results['cluster'],
        labels={'x':'PC1','y':'PC2','color':'Cluster'},
        title=f'PCA projection (k={n_clusters})'
    )
    if centers is not None:
        cent_pca = pca.transform(centers)
        fig.add_scatter(x=cent_pca[:,0], y=cent_pca[:,1], mode='markers', marker=dict(symbol='x', size=12, color='black'), name='Centroids')

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    st.subheader('Cluster summary')
    sizes = data_results['cluster'].value_counts().sort_index()
    st.table(pd.DataFrame({'cluster':sizes.index, 'size':sizes.values}).reset_index(drop=True))

    if hasattr(model, 'inertia_'):
        st.write('Inertia:', float(model.inertia_))
    if n_clusters >= 2:
        try:
            sil = silhouette_score(X_for_model, labels)
            st.write('Silhouette score:', float(sil))
        except Exception:
            pass

    # Show cluster centers (if available). If scaled, show centers in scaled units.
    if centers is not None:
        centers_df = pd.DataFrame(centers, columns=features)
        if scaler is not None:
            # inverse transform only if we scaled using our scaler
            try:
                orig_centers = scaler.inverse_transform(centers)
                centers_df = pd.DataFrame(orig_centers, columns=features)
                st.write('Cluster centers (approximate original scale)')
            except Exception:
                st.write('Cluster centers (scaled units)')
        else:
            st.write('Cluster centers (units in dataset)')
        st.dataframe(centers_df)

    st.subheader('Data sample with cluster labels')
    st.dataframe(data_results.head(200))

    csv = data_results.to_csv(index=False).encode('utf-8')
    st.download_button('Download labeled data (CSV)', data=csv, file_name='clustered_data.csv', mime='text/csv')

else:
    st.info('Adjust controls and click "Run clustering" to compute clusters.')
