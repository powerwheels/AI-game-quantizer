import json
import os
import gzip

#copy the path of the model you want to "quantize" and the one you want it to output to
file_to_open = r""
file_to_save = r""
allowed_values = [-1.0, -0.5, 0.0, 0.5, 1.0] #You can change the allowed_values list to include any set of quantized weights you want.

def quantize_value(value, target_weights):
    if not isinstance(value, (float, int)):
        return value
    return min(target_weights, key=lambda x: abs(x - value))
def process_nested_structure(data, target_weights):
    if isinstance(data, list):
        return [process_nested_structure(item, target_weights) for item in data]
    elif isinstance(data, dict):
        return {k: process_nested_structure(v, target_weights) for k, v in data.items()}
    else:
        return quantize_value(data, target_weights)

def run_quantization(input_path, output_path, target_weights):
    raw_content = None
    
    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            raw_content = f.read()
            print("Detected GZip compression. Decompressing...")
    except:
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(input_path, 'r', encoding=enc) as f:
                    raw_content = f.read()
                    print(f"Successfully read file using {enc} encoding.")
                    break
            except UnicodeDecodeError:
                continue
    if raw_content is None:
        print("Error: Could not read the file. It might be a different binary format.")
        return
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"Error: File read successfully, but it's not valid JSON. {e}")
        return
    if "NeuralNetwork" in data:
        nn = data["NeuralNetwork"]
        if "Weights" in nn:
            nn["Weights"] = process_nested_structure(nn["Weights"], target_weights)
        if "Biases" in nn:
            nn["Biases"] = process_nested_structure(nn["Biases"], target_weights)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Done! Saved quantized file to: {output_path}")
if os.path.exists(file_to_open):
    run_quantization(file_to_open, file_to_save, allowed_values)
else:
    print(f"File not found: {file_to_open}")