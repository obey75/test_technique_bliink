# Pokemon types - multi-label image classification

Given 800 pokemon images with their types, this project aims at developing a web interface to perform predictions about Pokemon types from images.

## Test the API

Open a terminal window at the root of the project and follow one by one these instructions :
```
.venv/bin/activate
pip install -r requirements.txt
python api/app.py
```

Open a web browser and go to `http://localhost:5000/`

On the interface, you can either :
- perform a prediction about the type of one of the 800 items given at the beginning : enter the pokemon name
- perform a prediction about the type of a Pokemon you found on the web : enter the url of the Pokemon image

## Analysis of the subject

Due to the small size of the dataset (800 images and 18 classes), we better use a pre-trained model.
<br>MobileNetV2 seems to be a good choice because of its lightness, which fits to our limited calculation capacities and the risk of overfitting our small dataset.
<br>We will not train the low layers of the model to avoid overfitting.
<br>To enhance the performance, we will implement different techniques : data augmentation, regularization and cross-validation.

## Preprocessing

To access the notebook dealing about data preprocessing, enter this command in the terminal at the root :
```
jupyter-notebook models\preprocessing.ipynb
```
The image size is (120, 120). Hence, the most appropriate size to train the model is (128, 128).
<br><br>
The original shape of the CSV file :
<br><br>
Name	Type1	Type2
<br>
0	bulbasaur	Grass	Poison
<br><br>
Let's use `datagen.flow_from_dataframe` from `ImageDataGenerator`class of `Keras` library to generate easily image batches from a dataframe.
<br>In order to do that, let's change the dataframe extracted from the CSV to make it look like that :
<br><br>
Name	Type1	Type2	labels	path
<br>
0	bulbasaur	Grass	Poison	[Grass, Poison]	bulbasaur.png
<br><br>
We prepare also a `test_generator` which have to remain unchanged and untrained to be able to test our model accurately.

## First model

```
jupyter-notebook models\first_model.ipynb
```
- We use a MobileNetV2 pre-trained model on 'imagenet' without the original top layer, replaced fully-connected layer for 18 classes multi-label classification.
- An appropriate activation function is 'sigmoid' for multi-label classification.
- The layer training is deactivated to avoid overfitting and exceeding calculation time.

## Fine-tuning

```
jupyter-notebook models\fine-tuning.ipynb
```
- Data augementation : add transformations of the images (rotations, shifts, zooms ...) to increase the size of the training dataset(x3) and reduce overfitting<br>The result is a less accurate model on the training dataset but less overfitting.
<br>
- Cross-validation<br>Our implementation of the 5-fold cross-validation didn't improve the accuracy of the model on the test dataser
