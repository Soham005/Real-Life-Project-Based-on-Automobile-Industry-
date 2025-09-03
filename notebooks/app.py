import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import os
import plotly.graph_objects as go
import plotly.express as px

# Brand dataset paths
brand_files = {
    "Mahindra": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\mahindra_dataset.csv",
    "Honda": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\honda_dataset.csv",
    "Hyundai": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\hyundai_dataset.csv",
    "Kia": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\kia_dataset.csv",
    "Maruti_N": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\maruti_n_dataset.csv",
    "Skoda": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\skoda_dataset.csv",
    "Tata": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\tata_dataset.csv",
    "Volkswagen": r"C:\Users\smcho\OneDrive\Desktop\Project 101\Real-Life-Project-Based-on-Automobile-Industry-\data\vw_dataset.csv",
}

st.title("📊 Automobile Sales Forecasting & Game Theory App")

brands = list(brand_files.keys())

# Select two brands
brand1 = st.selectbox("Select Brand 1", brands)
brand2 = st.selectbox("Select Brand 2", [b for b in brands if b != brand1])

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

# Forecast next 5 years
future_years = np.arange(df1_yearly["SaleDate"].max() + 1, df1_yearly["SaleDate"].max() + 6)
future_t1 = np.arange(len(df1_yearly), len(df1_yearly) + 5).reshape(-1, 1)
future_t2 = np.arange(len(df2_yearly), len(df2_yearly) + 5).reshape(-1, 1)

forecast1 = model1.predict(future_t1).round().astype(int)
forecast2 = model2.predict(future_t2).round().astype(int)

fc1 = pd.DataFrame({"Year": future_years, "Forecast": forecast1})
fc2 = pd.DataFrame({"Year": future_years, "Forecast": forecast2})

# Show results
st.subheader(f"{brand1} Forecast")
st.dataframe(fc1)

st.subheader(f"{brand2} Forecast")
st.dataframe(fc2)

# -------------------- GAME THEORY --------------------
st.subheader("🎯 Game Theory Analysis")

# Total forecasted sales for 5 years
total1 = fc1["Forecast"].sum()
total2 = fc2["Forecast"].sum()

st.write(f"🔹 **{brand1} Total (5 yrs):** {total1:,}")
st.write(f"🔹 **{brand2} Total (5 yrs):** {total2:,}")

# Simple payoff comparison
if total1 > total2:
    conclusion = f"✅ {brand1} has stronger forecasted demand. Strategic move: {brand1} should consider expanding production capacity in India."
elif total2 > total1:
    conclusion = f"✅ {brand2} has stronger forecasted demand. Strategic move: {brand2} should consider expanding production capacity in India."
else:
    conclusion = f"⚖️ Both {brand1} and {brand2} show similar sales potential. A joint venture or cautious expansion is recommended."

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

fig.update_layout(title="📈 Sales Forecast Comparison",
                  xaxis_title="Year", yaxis_title="Sales Units")

st.plotly_chart(fig, use_container_width=True)

# Pie chart of market share
pie_fig = px.pie(values=[total1, total2], names=[brand1, brand2],
                 title="Market Share (Next 5 Years)")
st.plotly_chart(pie_fig, use_container_width=True)
