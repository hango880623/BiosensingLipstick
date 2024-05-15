from flask import Flask, request, jsonify
import os
from post_request import load_model, predict, pH_value

result_folder = './results/results_2024-01-26_07-28-04-p-cen'
model_path = os.path.join(result_folder, 'best_resnet18_0.001_50.pth')

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def prediction():
    model = load_model(model_path, model_type='resnet18')
    if request.method == 'POST':
        # Ensure files are included in the request
        if 'files' not in request.files:
            return jsonify({'error': 'No files included in the request'})
        
        # Get the files from the request
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'Fail to get the files'})
        
        # Perform prediction for each file
        predictions = []
        for file in files:
            file_data = file.read()
            prediction = predict(model, file_data)
            predictions.append(pH_value[prediction.item()])
        
        return jsonify({'predictions': predictions})
    
if __name__ == '__main__':
    app.run(debug=True)
