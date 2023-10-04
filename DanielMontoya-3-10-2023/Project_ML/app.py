import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = load_model("model.h5")

# Función para realizar una predicción de emoción
def predict_emotion(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = model.predict(img)
    emotion_labels = ["angry", "happy", "sad"]
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    return predicted_emotion

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_emotion = None

    if request.method == 'POST':
        # Verifica si se ha enviado un archivo
        if 'file' not in request.files:
            return render_template('index.html', error='No se ha seleccionado ningún archivo.')

        file = request.files['file']

        # Verifica si se ha seleccionado un archivo
        if file.filename == '':
            return render_template('index.html', error='No se ha seleccionado ningún archivo.')

        # Verifica si el archivo es una imagen
        if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
            # Guarda temporalmente el archivo
            file_path = 'uploaded_image.png'
            file.save(file_path)

            # Realiza una predicción con el modelo
            predicted_emotion = predict_emotion(file_path)

    return render_template('index.html', predicted_emotion=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)
