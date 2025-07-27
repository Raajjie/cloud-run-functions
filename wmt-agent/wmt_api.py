from flask import Flask, jsonify, request
from wmt_agent import run_workflow

app = Flask(__name__)

# Create Database
unit_readings = []
all_results = []
duplicates_found = []
conflicts_found = []
tao_logs = []

# Create Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to WMT API!"})

@app.route('/process', methods=['POST'])
def process_input():
    input_data = request.get_json()
    
    # Extract the text field from the JSON input
    if not input_data or 'text' not in input_data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text_input = input_data['text']
    result = run_workflow(text_input)

    duplicates = result['duplicates_found']
    duplicates_found.append(duplicates)

    conflicts = result['conflicts_found']
    conflicts_found.append(conflicts)

    logs = result['logs']
    tao_logs.append(logs)


    json_output = next(step for step in result["all_results"] if step['tool'] == 'data_processing')
    unit_readings.append(json_output['output'])

    return jsonify(json_output['output']), 201

@app.route('/readings', methods=['GET'])
def get_all_readings():
    return jsonify(unit_readings)

@app.route('/readings/<int:index>', methods=['GET'])
def get_reading(index):
    if 0 <= index < len(unit_readings):
        return jsonify(unit_readings[index])
    else:
        return jsonify({"error": "Index out of range"}), 404

@app.route('/readings', methods=['DELETE']) 
def delete_all_readings(): 
    unit_readings.clear()
    return jsonify({"message": "All readings deleted"}), 200
    
@app.route('/readings/<int:index>', methods=['DELETE'])
def delete_reading(index):
    if 0 <= index < len(unit_readings):
        unit_readings.pop(index)

@app.route('/readings/duplicates/<int:index>', methods=['GET'])
def get_duplicates(index):
    return jsonify(duplicates_found[index])

@app.route('/readings/conflicts/<int:index>', methods=['GET'])
def get_conflicts(index):
    return jsonify(conflicts_found[index])

@app.route('/readings/logs/<int:index>', methods=['GET'])
def get_logs(index):
    return jsonify(tao_logs[index])







if __name__ == "__main__":
    app.run(debug=True)