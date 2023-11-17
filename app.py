import pickle
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import json

app = Flask("image-match")

with open("saved_model/mnist_decision_tree.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess(image_file):
    with Image.open(image_file) as img:
        img = img.convert('L')
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        image_array = np.array(img, dtype=np.float64)
        image_array = image_array.reshape(1, -1)
        image_array *= (16 / 255) 
    return image_array

def predict_digit(image):
    prediction = model.predict(image)
    return prediction

@app.route('/predict', methods=['POST'])
def check_same_digit():
    try:
        print(request.files.keys())
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        processed_image1 = preprocess(image1)
        processed_image2 = preprocess(image2)
        digit1 = predict_digit(processed_image1)
        digit2 = predict_digit(processed_image2)
        
        same_digit = digit1[0] == digit2[0]
        
        return json.dumps({'prediction': bool(same_digit)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
