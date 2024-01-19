import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image


IMG_WIDTH, IMG_HEIGHT = 128,128
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'images', 'images')
df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'pokemon.csv'))


def merge_types(row): 
    t1 = row['Type1']
    t2 = row['Type2']
    if t2 is np.nan: 
        return [t1]
    return [t1,t2]
    

def find_image_path_with_name(_name):
    if os.path.exists(os.path.join(IMAGES_FOLDER, _name+'.png')):
        return _name + '.png'
    else:
        return _name + '.jpg'


def load_train_val_generator():
	df['labels'] = df.apply(lambda row: merge_types(row), axis=1)
	df['path'] = df.apply(lambda row: find_image_path_with_name(row['Name']), axis=1)
	train_df, temp_df = train_test_split(df, train_size=0.8, random_state=42)
	validation_df, test_df =  train_test_split(temp_df, test_size=0.5, random_state=42)
	datagen = ImageDataGenerator(rescale=1./255)
	train_generator = datagen.flow_from_dataframe(
	    dataframe=train_df,
	    directory=IMAGES_FOLDER,
	    x_col='path',
	    y_col='labels',
	    target_size=(IMG_WIDTH, IMG_HEIGHT),
	    #batch_size=?,
	    class_mode='categorical',
	    validate_filenames=True
	)
	validation_generator = datagen.flow_from_dataframe(
	    dataframe=validation_df,
	    directory=IMAGES_FOLDER,
	    x_col='path',
	    y_col='labels',
	    target_size=(IMG_WIDTH, IMG_HEIGHT),
	    #batch_size=?,
	    class_mode='categorical',
	    validate_filenames=True
	)
	test_generator = datagen.flow_from_dataframe(
	    dataframe=test_df,
	    directory=IMAGES_FOLDER,
	    x_col='path',
	    y_col='labels',
	    target_size=(IMG_WIDTH, IMG_HEIGHT),
	    #batch_size=?,
	    class_mode='categorical',
	    validate_filenames=True
	)
	return train_generator, validation_generator, test_generator