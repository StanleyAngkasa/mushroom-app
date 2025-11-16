import streamlit as st
import pandas as pd
import pickle

# Load model
data = pickle.load(open("mushroom_model.pkl", "rb"))
model = data["model"]
label_encoder = data["label_encoder"]
features = data["features"]

st.title("🍄 Mushroom Classification App")
st.write("Prediksi apakah jamur **edible** atau **poisonous** berdasarkan fitur kategori.")

# Input form
inputs = {}
for col in features:
    inputs[col] = st.text_input(f"{col} (kode kategori)")

# Convert to DF
df_input = pd.DataFrame([inputs])

if st.button("Predict"):
    try:
        pred = model.predict(df_input)[0]
        label = label_encoder.inverse_transform([pred])[0]

        if label == "edible":
            st.success("🌿 Hasil: Jamur **AMAN dimakan** (edible)")
        else:
            st.error("☠️ Hasil: Jamur **BERACUN** (poisonous)")

    except Exception as e:
        st.error("Terjadi error: " + str(e))
