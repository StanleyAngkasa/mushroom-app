import streamlit as st
import pandas as pd
import pickle

data = pickle.load(open("mushroom_model.pkl", "rb"))
model = data["model"]
label_encoder = data["label_encoder"]
features = data["features"]

df = pd.read_csv("mushrooms.csv")

feature_options = {
    col: sorted(df[col].unique())
    for col in features
}

st.title("🍄 Mushroom Classification App (Full Label Version)")
st.write("Pilih nilai fitur jamur dari dropdown di bawah:")

user_input = {}

for col in features:
    user_input[col] = st.selectbox(
        f"{col}",
        options=feature_options[col],
        index=0
    )

df_input = pd.DataFrame([user_input])

if st.button("Predict"):
    try:
        pred = model.predict(df_input)[0]
        label_short = label_encoder.inverse_transform([pred])[0]

        # Label lengkap
        if label_short == "edible":
            label_full = "Edible (Aman Dimakan)"
            st.success(f"🌿 Hasil: **{label_full}**")
        else:
            label_full = "Poisonous (Beracun)"
            st.error(f"☠️ Hasil: **{label_full}**")

    except Exception as e:
        st.error("Terjadi error: " + str(e))
