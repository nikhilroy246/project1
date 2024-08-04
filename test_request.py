import requests
import json

# Define the URL of the Flask API
url = 'http://127.0.0.1:5000/predict'

# Prepare the input data
data = {
    'features': [1.5, 2.3, 3.1]  # Replace with actual feature values
}

# Define headers
headers = {'Content-Type': 'application/json'}

# Send POST request to Flask API
response = requests.post(url, data=json.dumps(data), headers=headers)

# Print the response from the API
print(response.json())
