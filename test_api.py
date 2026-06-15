import requests

data = {
    "Diameter": 900,
    "Height": 1200,
    "Thickness": 3,
    "Fibre": 1.5,
    "Taper": 3,
    "Concrete": 30
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)

print(response.text)