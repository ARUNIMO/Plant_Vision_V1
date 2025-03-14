import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import numpy as np
import pandas as pd
import CNN

# Load CSV data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=device))
model.to(device)
model.eval()

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask App Initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction Function
def prediction(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension & send to device
    output = model(input_data)
    index = torch.argmax(output, dim=1).item()  # Get the predicted class
    return index

# Home Page
@app.route('/')
def home_page():
    return render_template('home.html')

# AI Engine Page
@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

# Mobile Device Page
@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

# Image Upload & Prediction
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded!", 400

        image = request.files['image']
        if image.filename == '':
            return "No selected file!", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(file_path)

        pred = prediction(file_path)
        title = disease_info.iloc[pred]['disease_name']
        description = disease_info.iloc[pred]['description']
        prevent = disease_info.iloc[pred]['Possible Steps']
        image_url = disease_info.iloc[pred]['image_url']
        supplement_name = supplement_info.iloc[pred]['supplement name']
        supplement_image_url = supplement_info.iloc[pred]['supplement image']
        supplement_buy_link = supplement_info.iloc[pred]['buy link']

        return render_template('submit.html', 
                               title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, 
                               sname=supplement_name, simage=supplement_image_url, 
                               buy_link=supplement_buy_link)

# Marketplace Page
@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), 
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
