import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="AML601 Housing", page_icon="🏠", layout="wide")
st.title("AML601 - California Housing")
st.write("Simple dashboard for the AML601 project")

@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    if "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": "MedHouseValue"})
    return df

df = load_data()
median_val = df["MedHouseValue"].median()

st.sidebar.header("Settings")

min_val = float(df["MedInc"].min())
max_val = float(df["MedInc"].max())
medinc_range = st.sidebar.slider("MedInc", min_val, max_val, (min_val, max_val))

t_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)

task = st.sidebar.selectbox("Task", ["Regression", "Classification", "Clustering"])

filtered = df[df["MedInc"].between(medinc_range[0], medinc_range[1])]

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Map")
    if len(filtered) == 0:
        st.warning("No data for this range")
    else:
        plotdata = filtered[["Latitude", "Longitude", "MedHouseValue", "MedInc"]].copy()
        plotdata["radius"] = plotdata["MedHouseValue"] * 1200
        plotdata["radius"] = plotdata["radius"].clip(200, 12000)

        plotdata["r"] = (plotdata["MedHouseValue"] * 50).clip(0, 255)
        plotdata["g"] = (180 - plotdata["MedHouseValue"] * 15).clip(0, 255)
        plotdata["b"] = 220

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=plotdata,
            get_position="[Longitude, Latitude]",
            get_radius="radius",
            get_fill_color="[r, g, b, 160]",
            pickable=True,
        )

        view = pdk.ViewState(latitude=37.5, longitude=-119.5, zoom=5.4)

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"html": "<b>Value:</b> {MedHouseValue}<br/><b>MedInc:</b> {MedInc}"},
        ))

with col2:
    st.subheader("Stats")
    st.metric("Rows", len(filtered))
    if len(filtered) > 0:
        st.metric("Avg house value", round(filtered["MedHouseValue"].mean(), 3))
        st.metric("Avg MedInc", round(filtered["MedInc"].mean(), 3))
    else:
        st.metric("Avg house value", "N/A")
        st.metric("Avg MedInc", "N/A")

    st.write("Top rows:")
    st.dataframe(filtered[["MedInc", "MedHouseValue", "Latitude", "Longitude"]].head(10))