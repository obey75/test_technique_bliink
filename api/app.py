import sys
import os
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, 'models', 'first_model.h5')
DATA_PATH = os.path.join(ROOT, 'data')
CSV_PATH = os.path.join(DATA_PATH, 'pokemon.csv')
IMG_PATH = os.path.join(DATA_PATH, 'images', 'images')
IMG_WIDTH, IMG_HEIGHT = 128,128
df = pd.read_csv(CSV_PATH)

sys.path.append(ROOT)
from models.model import load_train_val_generator, merge_types, find_image_path_with_name
CLASSES = list(load_train_val_generator()[0].class_indices)
df['labels'] = df.apply(lambda row: merge_types(row), axis=1)
df['path'] = df.apply(lambda row: find_image_path_with_name(row['Name']), axis=1)


app = Flask(__name__)
model = load_model(MODEL_PATH)

@app.route('/')
def index():
	return ('hello world')

@app.route('/predict', methods=['POST'])
def predict():
	request_data = request.get_json(force=True)
	pokemon_name = request_data["pokemon_name"]	
	pokemon_path = os.path.join(IMG_PATH, find_image_path_with_name(pokemon_name))

	if os.path.exists(pokemon_path):
		img = image.load_img(pokemon_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
		img_array = image.img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		img_array /= 255.0

		prediction = model.predict(img_array)
		predicted_labels = [CLASSES[i] for i, p in enumerate(prediction[0]) if p > 0.3]
		return predicted_labels

	else:
		return jsonify({'error': 'Invalid Pokemon name'})


if __name__ == '__main__':
	app.run(host='localhost', port=5000, debug=True)