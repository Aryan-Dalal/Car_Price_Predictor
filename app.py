from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ... (Model loading and label_map definitions remain the same) ...
# ------------------------------
# 1️⃣ Load Trained Models
# ------------------------------
lr = joblib.load(os.path.join("models", "lr_model.pkl"))
dt = joblib.load(os.path.join("models", "dt_model.pkl"))
rf = joblib.load(os.path.join("models", "rf_model.pkl"))
voting = joblib.load(os.path.join("models", "voting_model.pkl"))
multi_rf = joblib.load(os.path.join("models", "multi_rf_model.pkl"))

# ------------------------------
# 2️⃣ Define Features and Label Encodings
# ------------------------------
feature_cols = [
    'symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody',
    'drivewheel', 'enginetype', 'cylindernumber', 'horsepower',
    'peakrpm', 'citympg', 'highwaympg'
]

label_map = {
    'fueltype': {'gas': 0, 'diesel': 1},
    'aspiration': {'std': 0, 'turbo': 1},
    'doornumber': {'two': 0, 'four': 1},
    'carbody': {'sedan': 0, 'hatchback': 1, 'wagon': 2, 'hardtop': 3, 'convertible': 4},
    'drivewheel': {'fwd': 0, 'rwd': 1, '4wd': 2},
    'enginetype': {'ohc': 0, 'ohcf': 1, 'ohcv': 2, 'dohc': 3, 'rotor': 4, 'l': 5},
    'cylindernumber': {'two': 0, 'three': 1, 'four': 2, 'five': 3, 'six': 4, 'eight': 5, 'twelve': 6}
}

# ------------------------------
# 3️⃣ Flask Routes
# ------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = []
            for col in feature_cols:
                val = request.form.get(col)
                
                if not val:
                     return f"Error: Missing value for {col}" 
                     
                # Check for symboling score range (NEW SERVER-SIDE VALIDATION)
                if col == 'symboling':
                    try:
                        symboling_val = int(val)
                        if not (-3 <= symboling_val <= 3):
                            return "Error: Symboling score must be between -3 and 3."
                        val = symboling_val
                    except ValueError:
                        return "Error: Symboling score must be a valid integer."
                
                # Encode categorical features
                elif col in label_map:
                    if val not in label_map[col]:
                        return f"Error: Invalid selection for {col}."
                    val = label_map[col][val]
                
                # Convert other numeric features
                input_data.append(float(val))
            
            input_array = np.array([input_data])

            # Predictions from models
            price_lr = lr.predict(input_array)[0]
            price_dt = dt.predict(input_array)[0]
            price_rf = rf.predict(input_array)[0]
            price_voting = voting.predict(input_array)[0]

            # Average price
            price_avg = np.mean([price_lr, price_dt, price_rf, price_voting])

            # Performance Prediction
            perf_pred = multi_rf.predict(input_array)[0][1]

            return render_template(
                'result.html',
                price=round(price_avg, 0), 
                perf=round(perf_pred, 0)
            )

        except Exception as e:
            return f"An error occurred during prediction: {e}"

    return render_template('index.html')


# ------------------------------
# 4️⃣ Run Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)