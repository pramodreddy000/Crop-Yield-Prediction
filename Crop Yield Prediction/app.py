from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gather form data
    N = float(request.form['nitrogen'])
    P = float(request.form['phosphorous'])
    K = float(request.form['potassium'])
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Prepare input features
    input_features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    
    # Predict the crop
    prediction = model.predict(input_features)
    predicted_class = np.argmax(prediction)
    crops=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
 'rice', 'watermelon']

    
    # Get the crop name
    predicted_crop = crops[predicted_class]
    
    # Render result.html with the prediction
    return render_template('result.html', prediction=predicted_crop)

if __name__ == '__main__':
    app.run(debug=True)
