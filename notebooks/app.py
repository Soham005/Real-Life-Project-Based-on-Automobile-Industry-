import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import os
import plotly.graph_objects as go
import plotly.express as px

# Brand dataset paths
brand_files = {
    "Mahindra": r"Real-Life-Project-Based-on-Automobile-Industry-/data/mahindra_dataset.csv",
    "Honda": r"Real-Life-Project-Based-on-Automobile-Industry-/data/honda_dataset.csv",
    "Hyundai": r"Real-Life-Project-Based-on-Automobile-Industry-/data/hyundai_dataset.csv",
    "Kia": r"Real-Life-Project-Based-on-Automobile-Industry-/data/kia_dataset.csv",
    "Maruti": r"Real-Life-Project-Based-on-Automobile-Industry-/data/maruti_n_dataset.csv",
    "Skoda": r"Real-Life-Project-Based-on-Automobile-Industry-/data/skoda_dataset.csv",
    "Tata": r"Real-Life-Project-Based-on-Automobile-Industry-/data/tata_dataset.csv",
    "Volkswagen": r"Real-Life-Project-Based-on-Automobile-Industry-/data/vw_dataset.csv",
}

st.title("📊 Automobile Sales Forecasting & Game Theory App")

brands = list(brand_files.keys())

# Select two brands
brand1 = st.selectbox("Select Brand 1", brands)
brand2 = st.selectbox("Select Brand 2", [b for b in brands if b != brand1])

# Select forecast horizon (years)
forecast_years = st.slider("Select Forecast Horizon (years)", min_value=3, max_value=10, value=5)

# Load models
model_dir = r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\notebooks"
model1 = pk.load(open(os.path.join(model_dir, f"{brand1}_forecast.pkl"), "rb"))
model2 = pk.load(open(os.path.join(model_dir, f"{brand2}_forecast.pkl"), "rb"))

# Load data
df1 = pd.read_csv(brand_files[brand1])
df2 = pd.read_csv(brand_files[brand2])

df1["SaleDate"] = pd.to_datetime(df1["SaleDate"], dayfirst=True, errors="coerce")
df2["SaleDate"] = pd.to_datetime(df2["SaleDate"], dayfirst=True, errors="coerce")

df1_yearly = df1.groupby(df1["SaleDate"].dt.year)["SalesUnits"].sum().reset_index()
df2_yearly = df2.groupby(df2["SaleDate"].dt.year)["SalesUnits"].sum().reset_index()

# Forecast next N years (based on user selection)
future_years1 = np.arange(df1_yearly["SaleDate"].max() + 1, df1_yearly["SaleDate"].max() + forecast_years + 1)
future_years2 = np.arange(df2_yearly["SaleDate"].max() + 1, df2_yearly["SaleDate"].max() + forecast_years + 1)

future_t1 = np.arange(len(df1_yearly), len(df1_yearly) + forecast_years).reshape(-1, 1)
future_t2 = np.arange(len(df2_yearly), len(df2_yearly) + forecast_years).reshape(-1, 1)

forecast1 = model1.predict(future_t1).round().astype(int)
forecast2 = model2.predict(future_t2).round().astype(int)

fc1 = pd.DataFrame({"Year": future_years1, "Forecast": forecast1})
fc2 = pd.DataFrame({"Year": future_years2, "Forecast": forecast2})

# Show results
st.subheader(f"{brand1} Forecast (Next {forecast_years} Years)")
st.dataframe(fc1)

st.subheader(f"{brand2} Forecast (Next {forecast_years} Years)")
st.dataframe(fc2)

# -------------------- GAME THEORY --------------------
st.subheader("🎯 Game Theory Analysis")

# Total forecasted sales
total1 = fc1["Forecast"].sum()
total2 = fc2["Forecast"].sum()

st.write(f"🔹 **{brand1} Total ({forecast_years} yrs):** {total1:,}")
st.write(f"🔹 **{brand2} Total ({forecast_years} yrs):** {total2:,}")

if total1 > total2:
    conclusion = f"✅ {brand1} is expected to dominate. Strategic move: expand production capacity in India."
elif total2 > total1:
    conclusion = f"✅ {brand2} is expected to dominate. Strategic move: expand production capacity in India."
else:
    conclusion = f"⚖️ Both brands show similar potential → joint venture or cautious expansion recommended."

st.markdown(f"### 🏆 Conclusion: {conclusion}")

# -------------------- PLOTLY VISUALS --------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df1_yearly["SaleDate"], y=df1_yearly["SalesUnits"],
                         mode="lines+markers", name=f"{brand1} History"))
fig.add_trace(go.Scatter(x=fc1["Year"], y=fc1["Forecast"],
                         mode="lines+markers", name=f"{brand1} Forecast", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=df2_yearly["SaleDate"], y=df2_yearly["SalesUnits"],
                         mode="lines+markers", name=f"{brand2} History"))
fig.add_trace(go.Scatter(x=fc2["Year"], y=fc2["Forecast"],
                         mode="lines+markers", name=f"{brand2} Forecast", line=dict(dash="dash")))

fig.update_layout(title=f"📈 Sales Forecast Comparison (Next {forecast_years} Years)",
                  xaxis_title="Year", yaxis_title="Sales Units")

st.plotly_chart(fig, use_container_width=True)

# Market share pie chart
pie_fig = px.pie(values=[total1, total2], names=[brand1, brand2],
                 title=f"Market Share (Next {forecast_years} Years)")
st.plotly_chart(pie_fig, use_container_width=True)
