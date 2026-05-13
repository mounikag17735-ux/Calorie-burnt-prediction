from flask import Flask, render_template, request

import pickle
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    gender = int(request.form['gender'])

    age = float(request.form['age'])

    height = float(request.form['height'])

    weight = float(request.form['weight'])

    duration = float(request.form['duration'])

    heart_rate = float(request.form['heart_rate'])

    body_temp = float(request.form['body_temp'])

    # Create dataframe
    features = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp
    }])

    # Prediction
    prediction = model.predict(features)

    return render_template(
        'index.html',
        prediction_text=f'Calories Burned: {prediction[0]:.2f}'
    )


# Run app
if __name__ == "__main__":
    app.run(debug=True)