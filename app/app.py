from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'student_grade_predictor.joblib')
encoder_path = os.path.join(base_dir, 'models', 'label_encoders.joblib')
features_path = os.path.join(base_dir, 'models', 'feature_columns.joblib')

model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)
feature_columns = joblib.load(features_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {}

        for col in feature_columns:
            val = request.form.get(col)

            # If value is missing, set a fallback
            if val is None:
                if col in label_encoders:
                    val = label_encoders[col].classes_[0]  # Use first known class
                else:
                    val = 0  # For numeric, use 0

            # Encode categorical
            if col in label_encoders:
                try:
                    val = label_encoders[col].transform([val])[0]
                except ValueError:
                    val = 0  # Fallback for unseen labels
            else:
                val = float(val)

            user_input[col] = val

        input_array = [user_input[col] for col in feature_columns]
        prediction = model.predict([input_array])[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f'Predicted Final Grade (G3): {prediction}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
# Define categorical fields and options manually or based on label_encoders
categorical_fields = list(label_encoders.keys())
options = {field: list(le.classes_) for field, le in label_encoders.items()}

@app.route('/')
def home():
    return render_template('index.html', features=feature_columns, categorical_fields=categorical_fields, options=options)

if __name__ == '__main__':
    app.run(debug=True)
