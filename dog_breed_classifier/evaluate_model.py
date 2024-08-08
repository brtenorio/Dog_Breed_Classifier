from keras.models import load_model
from dog_breed_classifier.config import *
import os

def evaluate_model(test):
    """
    This function evaluates the model using the test set.
    The model is loaded from the file_name path.
    """
    # check the existance of the model file
    if os.path.isfile(file_name):
        model = load_model(file_name)
    else:
        raise Exception("model file not found")
    test.reset()
    eval = model.evaluate(test)
    return eval

if __name__ == "__main__":
    from dog_breed_classifier.data_generator import data_generator
    _, _, test_generator = data_generator()
    eval = evaluate_model(test_generator)
    print('Test loss: ', eval[0])
    print('Test Accuracy: ', eval[1])