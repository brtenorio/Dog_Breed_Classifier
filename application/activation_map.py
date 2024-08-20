import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow.keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.applications.resnet50 import preprocess_input

def activation_map(model0, img_obj):
    """ This function takes a model, and a image object
    and returns the class activation map for the image"""

    img = img_obj.resize((224, 224))

    # Create a new model that will output the activations of the desired intermediate layers
    layer_names = ['conv5_block3_3_conv', 'predictions']
    outputs_dict = dict([(name, model0.get_layer(name).output) for name in layer_names])
    model = Model(inputs=model0.input, outputs=outputs_dict)

    # Get the prediction layer weights
    W = model.get_layer('predictions').get_weights()[0] # (2048, num_classes)

    # Preprocess the image
    x = preprocess_input(np.expand_dims(img, axis=0))

    # features is a dictionary to capture the activations of the intermediate layers
    features = model.predict(x)

    fmaps = features["conv5_block3_3_conv"][0] # (7, 7, 2048)

    # Get the predicted class activation map
    probs = features["predictions"]
    pred = np.argmax(probs)

    weights = W[:, pred] # (2048,)

    cam = np.dot(fmaps, weights) # (7, 7)

    # Resize the CAM to the size of the input image
    # (32, 32) x (7, 7) = (224, 224)
    am = sp.ndimage.zoom(cam, (32, 32), order=1)

    # Plot the original image and the CAM
    fig = plt.figure(figsize=(4, 4))
    
    plt.imshow(img, alpha=0.8)  # Base image
    plt.imshow(am, cmap='jet', alpha=0.4)  # CAM overlay
    plt.axis('off')
    #plt.title('Class Activation Map')

    # Convert the figure to a canvas and then to a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convert the rendered figure to a NumPy array
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Optionally, close the plot if you don't want it displayed
    plt.close(fig)

    return img_array