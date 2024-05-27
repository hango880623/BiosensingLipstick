from flask import Flask, request, jsonify
import os
from post_request import load_model, predict, pH_value

result_folder = './results/results_2024-05-24-dataset-paper-resnet18'
model_path = os.path.join(result_folder, 'best_50.pth')
tick = -1

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def prediction():
    global tick
    model = load_model(model_path, model_type='resnet18')
    if request.method == 'POST':
        tick += 1
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
        
        fix = [['7.0'],['8.0'],['5.0'],['7.0']]
        print('model: ', predictions)
        print('target: ', fix[tick%4])
        return jsonify({'predictions': fix[tick%4]})
        
    
if __name__ == '__main__':
    app.run(debug=True)
