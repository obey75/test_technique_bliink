import requests
import json

pokemon_name = 'aggron'

url = 'http://localhost:5000/predict'

requested_data = json.dumps({'pokemon_name': pokemon_name})
response = requests.post(url, requested_data)

print(response.text)