import pickle
import requests
import json
import numpy as np

# Load test data
with open('./data/X_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('./data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Get the first input and expected output
input_data = x_test[0]
expected_output = y_test[0]

# Prepare the request payload
payload = {
    "inputs": [input_data.tolist()]
}

# Send the request to the API
api_url = "http://192.168.1.125/invocations"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(api_url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()

    # Get the prediction
    prediction = response.json()["predictions"][0]
    print("API Response:", prediction)

    # Compare with expected output
    if isinstance(expected_output, np.ndarray):
        expected_output = expected_output.tolist()

    if prediction == expected_output:
        print("✅ Prediction is correct!")
    else:
        print("❌ Prediction does not match expected output.")
        print("Expected:", expected_output)
        print("Got:", prediction)

except requests.exceptions.RequestException as e:
    print("Error calling API:", e)
