from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# Create a function to preprocess input data
def preprocess_input(data):
    # Standardize numerical features (similar to what you did before)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        form_data = request.form.to_dict()
        user_input = pd.DataFrame(form_data, index=[0])

        # Preprocess the user input
        processed_input = preprocess_input(user_input)

        # Make predictions using the model
        prediction = model.predict(processed_input)

        # You can customize the response message based on the prediction
        if prediction[0] == 1:
            result = "Fraudulent"
        else:
            result = "Legitimate"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
