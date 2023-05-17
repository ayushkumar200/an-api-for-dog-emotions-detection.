# an-api-for-dog-emotions-detection.
in this project , i have created an API using python flask and jsonyfy
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

app = Flask(__name__)
model = ResNet50(weights='imagenet')

def predict_emotion(image):
    img = cv2.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = model.predict(img)
    predicted_emotions = decode_predictions(predictions, top=3)[0]

    emotions = [emotion[1] for emotion in predicted_emotions]
    return emotions

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    emotions = predict_emotion(img)
    return jsonify({'emotions': emotions})

if __name__ == '__main__':
    app.run()
import requests

url = 'http://localhost:5000/predict'
image_file = 'path_to_image.jpg'

with open(image_file, 'rb') as file:
    files = {'image': file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    emotions = result['emotions']
    print(emotions)
else:
    print('Error:', response.json())
