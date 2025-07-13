import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
import os

# ------------------ STEP 1: LOAD & ENCODE DATA ------------------
print("Loading and preparing data...")
df = pd.read_excel("canteen_demand_data.xlsx", engine='openpyxl')
df_encoded = pd.get_dummies(df, columns=["Weather", "Food_Item"])
X = df_encoded.drop(columns=["Date", "Demand"])
y = df_encoded["Demand"]

# ------------------ STEP 2: TRAIN OR LOAD MODEL ------------------
print("Training or loading model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_path = "canteen_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded from disk.")
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Model trained and saved to disk.")

# ------------------ STEP 3: EVALUATE MODEL ------------------
print("Evaluating model...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ------------------ STEP 4: ACTUAL VS PREDICTED ------------------
print("Plotting actual vs predicted with regression line...")
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
y_pred_test = model.predict(X_test_reset)

plt.figure(figsize=(12, 6))
plt.scatter(y_test_reset[:100], y_pred_test[:100], alpha=0.6, label='Predicted vs Actual')
reg_line = LinearRegression()
reg_line.fit(np.array(y_test_reset[:100]).reshape(-1, 1), y_pred_test[:100])
line_y = reg_line.predict(np.array(y_test_reset[:100]).reshape(-1, 1))
plt.plot(y_test_reset[:100], line_y, color='red', label='Best Fit Line')
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Model Accuracy: Actual vs Predicted with Best Fit Line")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_with_regression.png")
plt.show(block=False)
plt.pause(10.0)
plt.close()

# ------------------ STEP 5: PREDICT FUTURE DEMAND ------------------
print("Predicting demand for a sample day...")
sample_input = {
    "Weekday": 2,  # Wednesday
    "Is_Weekend": 0,
    "Is_Festival": 0,
    "Attendance": 220,
    "Weather_Cloudy": 0,
    "Weather_Rainy": 0,
    "Weather_Sunny": 1,
}

food_columns = [col for col in X.columns if col.startswith("Food_Item_")]
batch_predictions = {}
for food in food_columns:
    row = sample_input.copy()
    for f in food_columns:
        row[f] = 1 if f == food else 0
    row_df = pd.DataFrame([row])
    pred = model.predict(row_df)[0]
    food_name = food.replace("Food_Item_", "")
    batch_predictions[food_name] = round(pred)
batch_predictions = dict(sorted(batch_predictions.items(), key=lambda item: item[1], reverse=True))

# ------------------ STEP 6: PLOT PREDICTED DEMAND ------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(batch_predictions.keys(), batch_predictions.values(), color='skyblue')
plt.title("Predicted Servings Needed for Each Food Item (Sample Day)")
plt.xlabel("Food Item")
plt.ylabel("Predicted Servings")
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(height), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig("predicted_demand_bargraph.png")
plt.show(block=False)
plt.pause(10.0)
plt.close()

# ------------------ STEP 7: HISTORICAL AVERAGES ------------------
print("\nPredicted Demand vs Historical Average:\n")
historical_averages = df.groupby("Food_Item")["Demand"].mean().to_dict()
suggestions = []
for item, pred in batch_predictions.items():
    avg = round(historical_averages.get(item, 0))
    diff = pred - avg
    line = f"{item} --> Predicted: {pred}, Historical Avg: {avg}"
    if diff > 0:
        line += f" | Prepare ~{diff} extra"
    elif diff < 0:
        line += f" | Reduce by ~{-diff}"
    else:
        line += " | Matches typical demand"
    print(line)
    suggestions.append(line)

# ------------------ STEP 8: NOVELTY STATEMENT ------------------
novelty = (
    "NOVELTY: Unlike traditional demand forecasting which uses only past demand values,\n"
    "this system considers external factors like weekday, weather, festival, and attendance\n"
    "to make more dynamic and adaptable predictions."
)
print("\n" + novelty)

# ------------------ STEP 9: EXPORT PANEL SUMMARY ------------------
summary_path = "panel_summary_metrics.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("MODEL EVALUATION:\n")
    f.write(f"Mean Absolute Error: {mae:.2f}\n")
    f.write(f"R^2 Score: {r2:.2f}\n\n")
    f.write("PREDICTION SUMMARY (SAMPLE DAY):\n")
    for line in suggestions:
        f.write(line + "\n")
    f.write("\n" + novelty + "\n")
print(f"\nSummary written to {summary_path}")
