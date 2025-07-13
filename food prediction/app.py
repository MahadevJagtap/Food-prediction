import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("canteen_model.pkl")

st.set_page_config(page_title="Canteen Demand Predictor", layout="wide")
st.title("🍽️ Canteen Demand Predictor")

st.markdown("""
This tool predicts the number of servings needed for each food item based on day, weather, attendance, and festivals.
It helps in reducing food waste and ensures efficient preparation.
""")

# --- Input Section ---
st.header("📥 Enter Details")

weekday = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
weekday_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(weekday)

attendance = st.slider("Expected Attendance", 50, 500, 220)
is_festival = st.checkbox("Is it a festival today?", value=False)
weather = st.radio("Weather Condition", ["Sunny", "Cloudy", "Rainy"])

# Derived input values
is_weekend = 1 if weekday in ["Saturday", "Sunday"] else 0

# Prepare input dict
input_data = {
    "Weekday": weekday_idx,
    "Is_Weekend": is_weekend,
    "Is_Festival": int(is_festival),
    "Attendance": attendance,
    "Weather_Cloudy": 1 if weather == "Cloudy" else 0,
    "Weather_Rainy": 1 if weather == "Rainy" else 0,
    "Weather_Sunny": 1 if weather == "Sunny" else 0,
}

# Predict Button
if st.button("🔍 Predict Demand"):
    food_columns = [col for col in model.feature_names_in_ if col.startswith("Food_Item_")]
    predictions = {}

    for food in food_columns:
        row = input_data.copy()
        for f in food_columns:
            row[f] = 1 if f == food else 0
        df_row = pd.DataFrame([row])
        pred = model.predict(df_row)[0]
        food_name = food.replace("Food_Item_", "")
        predictions[food_name] = round(pred)

    sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

    st.header("📊 Predicted Servings per Food Item")
    st.bar_chart(pd.Series(sorted_predictions))

    st.markdown("""
    ### 🧠 Model Novelty
    Unlike traditional models that only use past numbers, this predictor considers real-world conditions:
    - **Weather** (Sunny, Rainy, Cloudy)
    - **Weekday vs Weekend**
    - **Festival Days**
    - **Expected Attendance**

    This makes predictions smarter, more adaptive, and reduces waste or shortages.
    """)
