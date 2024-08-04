from keras.models import load_model
from plant_seedlings.config import *


def evaluate_model(test):
    # check the existance of the model file
    if os.path.isfile(file_name):
        model = load_model(file_name)
    else:
        raise Exception("model file not found")
    test.reset()
    eval = model.evaluate(test)
    return eval

if __name__ == "__main__":
    from data_generator import test_generator
    eval = evaluate_model(test_generator)
    print('Test loss: ', eval[0])
    print('Test Accuracy: ', eval[1])