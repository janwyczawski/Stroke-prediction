import requests

# first, run the docker file with 'docker run -p 5001:5000 logistic_regression_model'

url = "http://localhost:5001/predict"

input_data = {
    "features": [
        ['Male',29,0,0,'No','Private','Urban',207.58,22.8,'smokes'],
        ["Male", 67, 0, 1, "Yes", "Private", "Urban", 228.69, 36.6, "formerly smoked"],
        ["Male", 80, 0, 1, "Yes", "Private", "Rural", 105.92, 32.5, "never smoked"],
        ["Female", 49, 0, 0, "Yes", "Private", "Urban", 171.23, 34.4, "smokes"],
        ["Female", 79, 1, 0, "Yes", "Self-employed", "Rural", 174.12, 24, "never smoked"],
        ["Male", 81, 0, 0, "Yes", "Private", "Urban", 186.21, 29, "formerly smoked"],
        ["Male", 74, 1, 1, "Yes", "Private", "Rural", 70.09, 27.4, "never smoked"],
        ["Female", 69, 0, 0, "No", "Private", "Urban", 94.39, 22.8, "never smoked"],
        ["Female", 78, 0, 0, "Yes", "Private", "Urban", 58.57, 24.2, "Unknown"]
    ]
}

response = requests.post(url, json=input_data)

if response.status_code == 200:
    predictions = response.json()
    print("Predictions:", predictions)
else:
    print(f"Error: {response.status_code} - {response.text}")