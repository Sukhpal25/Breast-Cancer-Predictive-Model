import joblib
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        input_features = [float(request.form.get(feature)) for feature in request.form]
        
        # Perform prediction
        prediction = model.predict([input_features])[0]
        
        # Redirect based on the prediction
        if prediction == 0:  # Benign
            return redirect(url_for('result', prediction='benign'))
        else:  # Malignant
            return redirect(url_for('result', prediction='malignant'))
    except Exception as e:
        return str(e)

@app.route('/result/<prediction>')
def result(prediction):
    if prediction == 'benign':
        return render_template('benign.html')
    elif prediction == 'malignant':
        return render_template('malignant.html')
    else:
        return "Invalid prediction", 400

if __name__ == '__main__':
    app.run(debug=True)
