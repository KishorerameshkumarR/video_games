from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model, label encoders, and scaler
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from the form
        platform = request.form['platform']
        genre = request.form['genre']
        na_sales = float(request.form['na_sales'])
        eu_sales = float(request.form['eu_sales'])
        jp_sales = float(request.form['jp_sales'])
        other_sales = float(request.form['other_sales'])

        # Encode categorical features
        platform_encoded = label_encoders['Platform'].transform([platform])[0]
        genre_encoded = label_encoders['Genre'].transform([genre])[0]

        # Prepare features for prediction
        features = np.array([[platform_encoded, genre_encoded, na_sales, eu_sales, jp_sales, other_sales]])
        features_scaled = scaler.transform(features)

        # Predict global sales
        prediction = model.predict(features_scaled)
        predicted_sales = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"Predicted Global Sales: {predicted_sales} million")

    except ValueError as e:
        return render_template('index.html', prediction_text=f"Error: {e}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Unexpected error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
