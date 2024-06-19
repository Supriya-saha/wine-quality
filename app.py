from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from request body (JSON format)
        features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                    'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        data = [request.json.get(feature, 0.0) for feature in features]
        final_features = np.array([data])
        
        # Make prediction
        prediction = model.predict(final_features)
        
        output = 'Good Quality' if prediction == 1 else 'Bad Quality'
        
        return {'prediction_text': f'Wine Quality: {output}'}

# Vercel expects to export a Flask app object as 'app'
app = app
