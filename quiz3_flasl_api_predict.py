import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask("quiz3")


model = load_model('my_model.keras')

def predict_digit(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    predicted_digit = prediction.argmax()
    return predicted_digit

@app.route('/check_same_digit', methods=['POST'])
def check_same_digit():
    try:
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        digit1 = predict_digit(image1)
        digit2 = predict_digit(image2)
        
        same_digit = digit1 == digit2
        
        return jsonify({'same_digit': same_digit})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
