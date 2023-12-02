import joblib

def load_model(model_type):
    if model_type == 'svm':
        return joblib.load('/mnt/d/Mtech_Practice/MLOps/Lectures/hand_written_digits_recognition/m22aie244_svm_rbf.joblib')
    elif model_type == 'lr':
        return joblib.load('/mnt/d/Mtech_Practice/MLOps/Lectures/hand_written_digits_recognition/m22aie244_lr_liblinear.joblib')
    elif model_type == 'tree':
        return joblib.load('m22aie244_tree_entropy_10.joblib')
    else:
        raise ValueError("Invalid model type")
    

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    # Load the model based on the URL
    model = load_model(model_type)

    # Extract features from request data
    data = request.get_json()
    features = data['features']

    # Make a prediction
    prediction = model.predict([features])

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

