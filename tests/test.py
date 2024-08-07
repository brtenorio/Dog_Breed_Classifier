import pytest 
from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

def test_model():
    """
    This function tests the model by evaluating it on an image of a Great Dane.
    """

    # Load the image
    test_file = 'tests/test_images/great_dane.jpg'
    image= Image.open(test_file)

    # Instantiate ImageDataGenerator to perform pre-processing on the loaded image
    image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    image_resize = 224
    image_resized = image.resize((image_resize, image_resize))
    image_resized = np.array(image_resized) # as numpy array

	# use image_generator to transform the image_transformed
    x = np.expand_dims(image_resized, axis=0)
    img_transformed = image_generator.flow(x)

    file_name = "saved_models/model.h5"
    if os.path.isfile(file_name):
        pass
    else:
        raise Exception("model not found!")
    
    #load the model
    model = load_model(file_name)

    # Evaluate on test_generator
    eval = model.predict(img_transformed)

    # predict the class. The model should predict the class of the image as 29, 
    # which is the class of the Great Dane.
    pred = np.argmax(eval,axis=-1) 

    #print('Test val: ', pred[0])

    assert pred[0] == 29

