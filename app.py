import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Energy Price Prediction Dashboard", layout="wide")

st.title("Energy Price Prediction Dashboard")
st.write("This dashboard predicts Germany energy price using Linear Regression and Random Forest.")

@st.cache_data
def load_data():
    df = pd.read_csv("electricity_dah_prices.csv")

    df.columns = df.columns.str.strip().str.lower().str.replace("\t", "", regex=False)

    df.rename(columns={
        "france": "france_energy_price",
        "italy": "italy_energy_price",
        "belgium": "belgium_energy_price",
        "spain": "spain_energy_price",
        "uk": "uk_energy_price",
        "germany": "germany_energy_price"
    }, inplace=True)

    df = df.dropna().copy()
    df["date"] = pd.to_datetime(df["date"])

   df["hour"] = df["hour"].astype(str).str.extract(r"(\d+)", expand=False)
df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
df = df.dropna(subset=["hour"]).copy()
df["hour"] = df["hour"].astype(int)

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year

    df["price_lag_1"] = df["germany_energy_price"].shift(1)
    df["price_lag_24"] = df["germany_energy_price"].shift(24)

    df = df.dropna().copy()
    return df

df = load_data()

country_features = [
    "france_energy_price",
    "italy_energy_price",
    "spain_energy_price",
    "uk_energy_price"
]

country_features = [col for col in country_features if col in df.columns]

features = [
    "hour",
    "day",
    "month",
    "dayofweek",
    "year",
    "price_lag_1",
    "price_lag_24"
] + country_features

X = df[features]
y = df["germany_energy_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [lr_rmse, rf_rmse],
        "R2 Score": [lr_r2, rf_r2]
    })
    st.dataframe(comparison, use_container_width=True)

with col2:
    st.subheader("Features Used")
    st.write(features)

st.subheader("Actual vs Predicted (Random Forest)")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_test.values[:100], label="Actual")
ax.plot(rf_pred[:100], label="Random Forest Predicted")
ax.set_title("Actual vs Predicted Germany Energy Price")
ax.set_xlabel("Index")
ax.set_ylabel("Germany Energy Price")
ax.legend()
st.pyplot(fig)

st.subheader("Actual vs Predicted (Linear Regression)")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(y_test.values[:100], label="Actual")
ax2.plot(lr_pred[:100], label="Linear Regression Predicted")
ax2.set_title("Actual vs Predicted Germany Energy Price")
ax2.set_xlabel("Index")
ax2.set_ylabel("Germany Energy Price")
ax2.legend()
st.pyplot(fig2)

st.subheader("Feature Importance (Random Forest)")
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.barh(importance.index, importance.values)
ax3.set_title("Feature Importance")
ax3.set_xlabel("Importance Score")
st.pyplot(fig3)

st.subheader("Try a Prediction")

c1, c2, c3 = st.columns(3)

with c1:
    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    dayofweek = st.slider("Day of Week", 0, 6, 2)

with c2:
    year = st.number_input("Year", min_value=int(df["year"].min()), max_value=int(df["year"].max()), value=int(df["year"].max()))
    price_lag_1 = st.number_input("Previous Hour Germany Price", value=float(df["germany_energy_price"].iloc[-1]))
    price_lag_24 = st.number_input("Germany Price 24 Hours Ago", value=float(df["germany_energy_price"].iloc[-24]))

with c3:
    france_price = st.number_input("France Energy Price", value=float(df["france_energy_price"].iloc[-1]) if "france_energy_price" in df.columns else 0.0)
    italy_price = st.number_input("Italy Energy Price", value=float(df["italy_energy_price"].iloc[-1]) if "italy_energy_price" in df.columns else 0.0)
    spain_price = st.number_input("Spain Energy Price", value=float(df["spain_energy_price"].iloc[-1]) if "spain_energy_price" in df.columns else 0.0)
    uk_price = st.number_input("UK Energy Price", value=float(df["uk_energy_price"].iloc[-1]) if "uk_energy_price" in df.columns else 0.0)

input_data = {
    "hour": hour,
    "day": day,
    "month": month,
    "dayofweek": dayofweek,
    "year": year,
    "price_lag_1": price_lag_1,
    "price_lag_24": price_lag_24
}

if "france_energy_price" in features:
    input_data["france_energy_price"] = france_price
if "italy_energy_price" in features:
    input_data["italy_energy_price"] = italy_price
if "spain_energy_price" in features:
    input_data["spain_energy_price"] = spain_price
if "uk_energy_price" in features:
    input_data["uk_energy_price"] = uk_price

input_df = pd.DataFrame([input_data])[features]

prediction = rf.predict(input_df)[0]

st.success(f"Predicted Germany Energy Price: {prediction:.2f}")
