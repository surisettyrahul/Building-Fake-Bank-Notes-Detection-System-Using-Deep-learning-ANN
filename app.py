from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model



# Now you can use this model to make predictions

# Initialize the Flask app
app = Flask(__name__)

# Load the model and scaler using pickle
# Load the model
model = load_model('models/model.h5')
with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Define a function for making predictions
def make_prediction(input_data):
    # Preprocess input data (apply scaling)
    input_data_scaled = scaler.transform(input_data)  # Use transform instead of fit_transform

    # Use the trained model to predict the class
    predictions = model.predict(input_data_scaled)

    # Convert prediction to binary (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int)

    return predicted_classes


# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Define the route for prediction
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get form data
        VWTI = float(request.form['VWTI'])
        SWTI = float(request.form['SWTI'])
        CWTI = float(request.form['CWTI'])
        EI = float(request.form['EI'])

        # Prepare input data for prediction
        input_data = np.array([[VWTI, SWTI, CWTI, EI]])

        # Get the prediction
        result = make_prediction(input_data)
        print(result)
        if result[0] == 1:
            output = "real"
        else:
            output = "fake"
        print(output)
        # Pass the result to the template
        return render_template('index.html',prediction=output)  # result[0] gives the first element in array (0 or 1)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)