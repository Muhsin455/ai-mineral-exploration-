
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="MineralAI", page_icon="⛰️")

st.title("⛰️ MineralAI")
st.subheader("AI for Mineral Exploration")
st.write("Satellite + Seismic Data Analysis")

st.warning("Demo only: sample data, not real geological findings.")

data = pd.DataFrame({
    "spectral": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9,
                 0.15, 0.25, 0.75, 0.85, 0.4, 0.65],
    "seismic": [8, 12, 15, 35, 40, 45, 10, 14, 38, 42, 18, 30],
    "mineral": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
})

X = data[["spectral", "seismic"]]
y = data["mineral"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.success("Demo model trained successfully!")

spectral = st.slider("Spectral index", 0.0, 1.0, 0.6)
seismic = st.slider("Seismic value", 0, 50, 25)

if st.button("Predict Mineral Potential"):
    result = model.predict(pd.DataFrame({
        "spectral": [spectral],
        "seismic": [seismic]
    }))[0]

    if result == 1:
        st.info("Demo prediction: Higher potential")
    else:
        st.info("Demo prediction: Lower potential")

    st.caption("This prediction uses illustrative sample data.")

