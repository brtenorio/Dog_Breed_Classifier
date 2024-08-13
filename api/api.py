import os
from typing import Union
from fastapi import FastAPI, Response, UploadFile

def process_image(upload):
    """
    This function processes the image file-object uploaded by the user and returns 
    the predicted class of the image as a string.
    """
    import numpy as np
    import io
    from PIL import Image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.resnet50 import preprocess_input
    from keras.models import load_model

    # Open the image with PIL by reading the uploaded image as bytes
    image= Image.open(io.BytesIO(upload))

    # Instantiate ImageDataGenerator to perform pre-processing on the loaded image
    image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    image_resize = 224 # the target size 
    image_resized = image.resize((image_resize, image_resize))
    image_resized = np.array(image_resized)

	# use image_generator to transform the image_transformed
    x = np.expand_dims(image_resized, axis=0)
    img_transformed = image_generator.flow(x)

    # Get the path to the model file and load it
    file_name = "saved_models/model.h5"
    file_name = os.path.normpath(file_name)
    if os.path.isfile(file_name):
        pass
    else:
        raise Exception("model not found!")
    model = load_model(file_name)

    # Predict the image using the model
    eval = model.predict(img_transformed)
    pred = np.argmax(eval,axis=-1)

    # Read external file class_names.txt as dictionary class_names
    class_names = {}
    class_names_file = os.path.join(os.path.dirname(__file__) , 'class_names.txt')

    # Read the class names from the class_names.txt file
    with open(class_names_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            class_names[int(key)] = value.strip()

    # Get the predicted class name
    ind = int(pred)
    prediction = class_names[ind]
    prediction = prediction.replace("_", " ")
    return prediction

app = FastAPI()

@app.get("/")
async def root():
    """This endpoint returns the home page of the API."""
    return Response("<h1> Dog Breed Classifier </h1>")

@app.post("/upload")
async def upload_image(file: Union[UploadFile, None] = None): 
    """
    This endpoint allows the user to upload an image file-object 
    and returns the predicted dog breed.
    """
    if not file:
        return {"message": "No upload file sent"}
    else:
        # Read the uploaded image file as bytes
        upload = await file.read()

        # Process the uploaded image
        dog = process_image(upload)

        # Return the predicted dog breed
        return {"breed": dog}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)