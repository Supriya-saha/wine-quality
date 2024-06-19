from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                    'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        data = [float(request.form[feature]) for feature in features]
        final_features = np.array([data])
        
        # Make prediction
        prediction = model.predict(final_features)
        
        output = 'Good Quality' if prediction == 1 else 'Bad Quality'
        
        return render_template('index.html', prediction_text=f'Wine Quality: {output}')

if __name__ == "__main__":
    app.run(debug=True)
