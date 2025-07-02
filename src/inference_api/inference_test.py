import pickle
import requests
import json
import numpy as np
import pandas as pd

# Load test data
with open('./data/X_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('./data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

input_data = x_test[0]
expected_output = y_test[0]

if isinstance(input_data, np.ndarray) and input_data.ndim == 1:
    input_data = np.expand_dims(input_data, axis=0)

# Convert to Pandas DataFrame for MLflow pyfunc input.
df_input = pd.DataFrame(input_data.astype(np.float32))

# Prepare the request payload as a dictionary with 'dataframe_split' key
payload_dict = {
    "dataframe_split": {
        "columns": df_input.columns.tolist(),
        "data": df_input.values.tolist()
    }
}
payload = json.dumps(payload_dict)

# Send the request to the API
api_url = "http://192.168.1.125/invocations"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(api_url, data=payload, headers=headers)
    response.raise_for_status()

    # Get the raw prediction output from the first sample
    raw_prediction_output = response.json()["predictions"][0]

    prediction = None
    if isinstance(raw_prediction_output, dict):
        # PyTorch model output: {'0': score_0, '1': score_1}
        class_scores = {int(k): v for k, v in raw_prediction_output.items()}
        prediction = max(class_scores, key=class_scores.get)
    elif isinstance(raw_prediction_output, (list, np.ndarray)):
        # Logistic Regression (or similar) output:
        # Could be probabilities (e.g., [0.2, 0.8]) or a single-element list ([1])
        if len(raw_prediction_output) == 1 and isinstance(raw_prediction_output[0], (int, float)):
            # It's likely a direct single class prediction like [1]
            prediction = int(raw_prediction_output[0])
        else:
            # It's likely probabilities, find the index of the max
            prediction = int(np.argmax(raw_prediction_output))
    elif isinstance(raw_prediction_output, (int, float)):
        # Direct scalar prediction (e.g., 1 or 0.7 if it's a probability needing rounding)
        prediction = int(round(raw_prediction_output)) # Round if it could be a probability

    print("API Response (raw):", raw_prediction_output)
    print("Predicted Class:", prediction)

    if isinstance(expected_output, np.ndarray):
        if expected_output.ndim == 0:
            expected_output = expected_output.item()
        elif expected_output.ndim == 1 and expected_output.size == 1:
            expected_output = expected_output[0].item()
        else:
            expected_output = expected_output.tolist()

    expected_output = int(expected_output)

    if prediction == expected_output:
        print("✅ Prediction is correct!")
    else:
        print("❌ Prediction does not match expected output.")
        print("Expected:", expected_output)
        print("Got:", prediction)

except requests.exceptions.RequestException as e:
    if hasattr(e, 'response') and e.response is not None:
        print(f"Error calling API: {e.response.status_code} {e.response.reason}")
        print("Response Body:", e.response.text)
    else:
        print("Error calling API:", e)
