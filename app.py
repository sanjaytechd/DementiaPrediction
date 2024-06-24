from flask import Flask, render_template, request, send_file, session, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import secrets

app = Flask(__name__)
secret_key = secrets.token_hex(16)
app.secret_key = secret_key

model_path = 'C:/DementiaPrediction/model.h5'
svm_model_path = 'C:/DementiaPrediction/svm2_model.pkl'


loaded_model = load_model(model_path)
loaded_svm = joblib.load(svm_model_path)


image_height, image_width = 128, 128


def preprocess_new_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_height, image_width))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def get_class_label(class_index):
    class_labels = ['NonDemented', 'MildDemented', 'VeryMildDemented', 'ModerateDemented']
    return class_labels[class_index]

@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # You should have your own logic to validate the user
        if username == 'admin' and password == '123456':
            session['username'] = username
            return redirect(url_for('home'))
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/logout')
def logout():
    
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
def predict():
    user_input = []
    for feature_name in ["M/F", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]:
        value = float(request.form[feature_name])
        user_input.append(value)
    user_input_array = np.array(user_input).reshape(1, -1)

    
    image_file = request.files['image']
    image_path = "static/" + image_file.filename
    image_file.save(image_path)

    
    preprocessed_image = preprocess_new_image(image_path)
    predictions = loaded_model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_label = get_class_label(predicted_class_index)
    model_result = get_class_label(predicted_class_index)

    
    prediction = loaded_svm.predict(user_input_array)
    if prediction[0] == 1:
        svm_result = 'Demented'
    elif prediction[0] == 0:
        svm_result = 'Non-Demented'
    
    if prediction[0] == 1 and predicted_class_index != 0:
        final_result = 'Demented'
    elif prediction[0] == 0 and predicted_class_index == 0:
        final_result = 'Non-Demented'
    elif prediction[0] == 0 and predicted_class_index == 3:
        final_result = 'Demented'
    else:
        final_result = 'Further Test Recommended'

    return render_template('result.html', image_path=image_path, 
                           svm_result=svm_result, model_result=model_result,
                           final_result=final_result)
@app.route('/generate_report', methods=['POST'])
def generate_report():
    
    patient_name = request.form['patient_name']
    patient_id = request.form['patient_id']
    svm_result = request.form['svm_result']
    model_result = request.form['model_result']
    final_result = request.form['final_result']
    now = datetime.now()
    current_date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    
    pdf_filename = 'report.pdf'
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph('DEMENTIA REPORT', styles['Title']))
    content.append(Spacer(1, 12))
    content.append(Paragraph('----------------------------------------------------------------------------------------------------------------------------------------'))

    content.append(Paragraph('Patient Details:', styles['Heading2']))
    content.append(Paragraph(f'Name: {patient_name}', styles['Normal']))
    content.append(Paragraph(f'ID: {patient_id}', styles['Normal']))
    content.append(Paragraph(f'Date and Time: {current_date_time}', styles['Normal'])) 
    content.append(Spacer(1, 12))
    content.append(Paragraph('----------------------------------------------------------------------------------------------------------------------------------------'))
    content.append(Paragraph('Findings:', styles['Heading2']))
    content.append(Paragraph(f'TEST 1 :                             {svm_result}', styles['Normal']))
    content.append(Paragraph(f'TEST 2 :                             {model_result}', styles['Normal']))
    content.append(Paragraph(f'Result: {final_result}', styles['Heading2']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f'NOTE',styles['Heading4']))
    content.append(Paragraph('We have provided the results of the analysis based on the information and images provided. Please note that this analysis is for informational purposes only and should not be considered a definitive diagnosis.', styles['Normal']))
    content.append(Paragraph('If the results indicate a possibility of dementia or any other health concern, we strongly advise you to consult with a qualified healthcare professional for further evaluation and guidance.', styles['Normal']))
    content.append(Paragraph('Your healthcare provider will be able to provide personalized recommendations, further diagnostic tests if necessary, and a comprehensive treatment plan based on your individual health status.', styles['Normal']))
    content.append(Paragraph('Remember, early detection and proper medical care are crucial for managing and addressing health conditions effectively.', styles['Normal']))
    
    content.append(Spacer(1, 100))  
    content.append(Paragraph('Signature of Lab Technician:', styles['Normal'])) 
    doc.build(content)

    
    return send_file(pdf_filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
