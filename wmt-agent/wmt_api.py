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

@app.route('/process', methods=['POST', 'PUT'])
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


    json_output = next((step for step in result["all_results"] if step['tool'] == 'data_processing'), None)
    if json_output is None:
        # Handle the error gracefully, e.g.:
        return {"Error": "Ambigious input found. Please try again."}
    unit_readings.append(json_output['output'])

    return jsonify(json_output['output']), 201

@app.route('/process/<int:index>', methods=['PUT'])
def update_process(index):
    input_data = request.get_json()

    if not input_data or 'text' not in input_data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    text_input = input_data['text']
    result = run_workflow(text_input)
    
    duplicates = result['duplicates_found']
    duplicates_found[index] = duplicates

    conflicts = result['conflicts_found']
    conflicts_found[index] = conflicts

    logs = result['logs']
    tao_logs[index] = logs
    
    json_output = next((step for step in result["all_results"] if step['tool'] == 'data_processing'), None)
    if json_output is None:
        # Handle the error gracefully, e.g.:
        return {"Error": "Ambigious input found. Please try again."}
    unit_readings[index] = json_output['output']

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
        reading = unit_readings.pop(index)

    return jsonify({"message": f"Reading {reading} deleted"}), 200


@app.route('/readings/<string:kind>', methods=['GET'])
def get_all_readings_item(kind):
    mapping = {
        'duplicates': duplicates_found,
        'conflicts': conflicts_found,
        'logs': tao_logs
    }
    if kind not in mapping:
        return jsonify({"error": "Invalid kind"}), 400
    return jsonify(mapping[kind])

def get_item_from_list(data_list, index, label):
    """Helper to safely get an item by index or return an error."""
    try:
        return jsonify(data_list[index])
    except IndexError:
        return jsonify({"error": f"{label} index {index} out of range"}), 404


@app.route('/readings/<string:kind>/<int:index>', methods=['GET'])
def get_reading_item(kind, index):
    mapping = {
        'duplicates': (duplicates_found, "Duplicates"),
        'conflicts': (conflicts_found, "Conflicts"),
        'logs': (tao_logs, "Logs")
    }
    if kind not in mapping:
        return jsonify({"error": "Invalid kind"}), 400
    data_list, label = mapping[kind]
    return get_item_from_list(data_list, index, label)
    


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)