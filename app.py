import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, silhouette_score
from sklearn.decomposition import PCA

# اعدادات الصفحة
st.set_page_config(page_title="AML601 Housing", page_icon="🏠", layout="wide")
st.title("AML601 - California Housing")
st.write("Simple dashboard for the AML601 project")

# تحميل البيانات
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    if "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": "MedHouseValue"})
    return df

df = load_data()
median_val = df["MedHouseValue"].median()

# الشريط الجانبي
st.sidebar.header("Settings")

min_med = float(df["MedInc"].min())
max_med = float(df["MedInc"].max())
medinc_range = st.sidebar.slider("MedInc", min_med, max_med, (min_med, max_med))

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)

task = st.sidebar.selectbox("Task", ["Regression", "Classification", "Clustering"])

# فلترة البيانات
filtered = df[df["MedInc"].between(medinc_range[0], medinc_range[1])]

# تخطيط الصفحة
col1, col2 = st.columns([1.2, 1])

# الخريطة على اليسار
with col1:
    st.subheader("Map")
    if len(filtered) == 0:
        st.warning("No data for this range")
    else:
        plotdata = filtered[["Latitude", "Longitude", "MedHouseValue", "MedInc"]].copy()
        plotdata["radius"] = plotdata["MedHouseValue"] * 1200
        plotdata["radius"] = plotdata["radius"].clip(200, 12000)

        # الالوان حسب القيمة
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

# الاحصائيات على اليمين
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

# قسم الموديلات
st.divider()
st.header(task)

X = df.drop(columns=["MedHouseValue"])
y = df["MedHouseValue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if task == "Regression":
    st.write("Comparing linear models")

    # Linear
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_test)

    # Ridge
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    pred_ridge = ridge.predict(X_test_s)

    # Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_s, y_train)
    pred_lasso = lasso.predict(X_test_s)

    # عرض النتائج
    results = pd.DataFrame({
        "Model": ["Linear", "Ridge", "Lasso"],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, pred_lin)),
            np.sqrt(mean_squared_error(y_test, pred_ridge)),
            np.sqrt(mean_squared_error(y_test, pred_lasso)),
        ],
        "R2": [
            r2_score(y_test, pred_lin),
            r2_score(y_test, pred_ridge),
            r2_score(y_test, pred_lasso),
        ],
    })
    st.dataframe(results.round(4))

    # رسم بسيط
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred_lin, s=3, alpha=0.3)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Linear Regression")
    st.pyplot(fig)

    # معاملات الموديل
    st.write("Coefficients:")
    coefs = pd.DataFrame({"Feature": X.columns, "Coef": lin.coef_})
    coefs = coefs.sort_values("Coef", key=abs, ascending=False)
    st.bar_chart(coefs.set_index("Feature"))

elif task == "Classification":
    st.write("Expensive (1) if value > median, else Affordable (0)")

    y_bin = (df["MedHouseValue"] > median_val).astype(int)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_bin, test_size=test_size, random_state=42
    )

    # Logistic
    scaler = StandardScaler()
    X_train_bs = scaler.fit_transform(X_train_b)
    X_test_bs = scaler.transform(X_test_b)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_bs, y_train_b)
    pred_log = logreg.predict(X_test_bs)

    # Decision tree
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train_b, y_train_b)
    pred_tree = tree.predict(X_test_b)

    results = pd.DataFrame({
        "Model": ["Logistic", "Decision Tree"],
        "Accuracy": [
            accuracy_score(y_test_b, pred_log),
            accuracy_score(y_test_b, pred_tree),
        ],
        "F1": [
            f1_score(y_test_b, pred_log),
            f1_score(y_test_b, pred_tree),
        ],
    })
    st.dataframe(results.round(4))

    st.write("Class counts:")
    counts = y_bin.value_counts()
    counts.index = ["Affordable", "Expensive"]
    st.bar_chart(counts)

elif task == "Clustering":
    st.write("KMeans on standardized features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA للرسم
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    k = st.slider("Number of clusters", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    st.metric("Silhouette", round(sil, 4))
    st.metric("Inertia", int(km.inertia_))

    c1, c2 = st.columns(2)

    with c1:
        st.write("PCA view")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=4)
        st.pyplot(fig)

    with c2:
        st.write("Map view")
        mapdata = df[["Latitude", "Longitude"]].copy()
        mapdata = mapdata.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        st.map(mapdata)

st.divider()
st.caption("AML601 project - Streamlit dashboard")