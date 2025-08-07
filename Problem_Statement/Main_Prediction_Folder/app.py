from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model and assets
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', geos=onehot_encoder_geo.categories_[0], genders=label_encoder_gender.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    geo = data['geography']
    gender = label_encoder_gender.transform([data['gender']])[0]

    input_df = pd.DataFrame({
        'CreditScore': [data['credit_score']],
        'Gender': [gender],
        'Age': [data['age']],
        'Tenure': [data['tenure']],
        'Balance': [data['balance']],
        'NumOfProducts': [data['num_of_products']],
        'HasCrCard': [data['has_cr_card']],
        'IsActiveMember': [data['is_active_member']],
        'EstimatedSalary': [data['estimated_salary']]
    })

    geo_encoded = onehot_encoder_geo.transform([[geo]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    full_input = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)
    full_input_scaled = scaler.transform(full_input)

    prediction = model.predict(full_input_scaled)[0][0]
    return jsonify({'churn_probability': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
