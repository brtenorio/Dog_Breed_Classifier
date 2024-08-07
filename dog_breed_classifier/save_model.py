import os
from keras.models import Model

def save_model(model):
    """"
    Save the model to a file in the saved_models directory:
    input: model object.
    """

    file_name = "saved_models/model.h5"
    if os.path.isdir("saved_models"):
        if os.path.isfile(file_name):
            os.remove(file_name)
            model.save(file_name)
            print("model updated")
        else:
            model.save(file_name)
            print("model created")
    else:
        raise Exception("saved_models directory not found")
    
if __name__ == "__main__":
    from get_model import get_model
    model = get_model()
    save_model(model)