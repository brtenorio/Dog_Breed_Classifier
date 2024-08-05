import pytest 
from dog_breed_classifier.evaluate_model import evaluate_model
from dog_breed_classifier.data_generator import data_generator

def test_model():
    _, _, test = data_generator()
    eval = evaluate_model(test)
    print('Test Accuracy: ', eval[1])
    try: 
        assert eval[1] > 0.60
    except:
        raise Exception("test failed. Try rebuilding the model.")

