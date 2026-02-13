import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Title
st.title("Access to Care Dataset Dashboard")
st.markdown("An interactive dashboard presenting key insights from the Access to Care dataset.")

# Load data
@st.cache_data

def load_data(path=None):
    # allow an environment variable or explicit path
    if path is None:
        path = st.secrets.get("DATA_PATH") if "DATA_PATH" in st.secrets else None
    if path is None:
        # default to local filename
        path = "Access_to_Care_Dataset.csv"
    # try absolute fallback locations
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        # try looking in parent directory or OneDrive path if provided
        alt = r"C:\Users\warre\OneDrive\Access_to_Care_Dataset.csv"
        if path != alt:
            try:
                return pd.read_csv(alt)
            except FileNotFoundError:
                pass
        raise

# attempt loading
try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "Dataset file not found. Either place 'Access_to_Care_Dataset.csv' in the app folder,\n" \
        "set DATA_PATH in Streamlit secrets, or ensure the file exists at C:\\Users\\warre\\OneDrive\\Access_to_Care_Dataset.csv."
    )
    st.stop()

# Data overview
st.header("Dataset Overview")
st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
st.dataframe(data.head())

# Clean data
clean = data[data['FLAG'].isna() & data['ESTIMATE'].notna()].copy()

# Sidebar controls
st.sidebar.header("Filters")
selected_topic = st.sidebar.multiselect("Select Topic", options=clean['TOPIC'].unique(), default=clean['TOPIC'].unique())
selected_subgroup = st.sidebar.multiselect("Select Subgroup", options=clean['SUBGROUP'].unique(), default=clean['SUBGROUP'].unique())
selected_classif = st.sidebar.multiselect("Select Classification", options=clean['CLASSIFICATION'].unique(), default=clean['CLASSIFICATION'].unique())

filtered = clean[
    clean['TOPIC'].isin(selected_topic) &
    clean['SUBGROUP'].isin(selected_subgroup) &
    clean['CLASSIFICATION'].isin(selected_classif)
]

st.header("Filtered Data")
st.write(f"Rows after filter: {filtered.shape[0]}")
st.dataframe(filtered.head())

# Visualizations
st.header("Visualizations")

# 1. Distributions by SUBGROUP
if not filtered.empty:
    fig1 = px.box(filtered, x='SUBGROUP', y='ESTIMATE', points='all', title='Estimate Distribution by Subgroup')
    st.plotly_chart(fig1, use_container_width=True)

# 2. Trends over time if TIME_PERIOD exists
if 'TIME_PERIOD' in filtered.columns:
    fig2 = px.line(
        filtered.groupby(['TIME_PERIOD'])['ESTIMATE'].mean().reset_index(),
        x='TIME_PERIOD', y='ESTIMATE', title='Average Estimate Over Time'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Trend by topic
    fig3 = px.line(
        filtered.groupby(['TIME_PERIOD','TOPIC'])['ESTIMATE'].mean().reset_index(),
        x='TIME_PERIOD', y='ESTIMATE', color='TOPIC', title='Trend by Topic'
    )
    st.plotly_chart(fig3, use_container_width=True)

# 3. Pivot heatmap: TOPIC vs SUBGROUP
pivot = filtered.pivot_table(values='ESTIMATE', index='TOPIC', columns='SUBGROUP', aggfunc='mean')
if not pivot.empty:
    fig4 = px.imshow(pivot, aspect='auto', title='Mean Estimate: Topic vs Subgroup')
    st.plotly_chart(fig4, use_container_width=True)

# 4. Summary statistics
st.header("Summary Statistics")
if 'TOPIC' in filtered.columns:
    stats = filtered.groupby('TOPIC')['ESTIMATE'].describe()
    st.dataframe(stats)

st.markdown("---")
st.markdown("*Dashboard generated with Streamlit.*")
