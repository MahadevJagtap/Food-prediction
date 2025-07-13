🍽️ Canteen Food Demand Predictor
This project is a machine learning-based system that predicts the number of servings needed for various food items in a canteen based on real-world parameters like:

📅 Day of the week

🌤️ Weather conditions

🎉 Festivals

👥 Expected attendance

The goal is to reduce food wastage and help canteen managers prepare the right quantity of food daily.

📌 Features
✅ Streamlit-based Web App Interface

📊 Interactive visual prediction of food demand

🧠 Model trained using Random Forest Regressor

📈 Evaluates model performance using MAE and R²

📤 Exports summary reports and graphs

🛠️ Tech Stack
Python

Pandas, NumPy, Matplotlib

Scikit-learn

Streamlit

Joblib

📂 Folder Structure
bash
Copy
Edit
├── app.py                      # Streamlit web app
├── canteen_predictor.py       # Model training and evaluation script
├── canteen_demand_data.xlsx   # Dataset for training
├── canteen_model.pkl          # Trained ML model (generated after training)
├── predicted_demand_bargraph.png
├── actual_vs_predicted_with_regression.png
├── panel_summary_metrics.txt  # Summary of evaluation
🚀 How to Run
1. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Or manually install:

bash
Copy
Edit
pip install pandas numpy matplotlib scikit-learn streamlit openpyxl joblib
2. Train the Model
bash
Copy
Edit
python canteen_predictor.py
3. Run the Web App
bash
Copy
Edit
streamlit run app.py
📊 Output Screenshots
Add screenshots of the Streamlit web app and prediction graphs here.

💡 Novelty
Unlike traditional demand prediction models, this system integrates contextual features like festivals, weekends, and weather to make smarter and adaptive predictions.

👤 Author
Mahadev Jagtap
3rd Year B.Tech CSE-AI&ML, Dayananda Sagar University
📧 [mmjagtap007@gmail.com]
💼 [https://github.com/MahadevJagtap]
