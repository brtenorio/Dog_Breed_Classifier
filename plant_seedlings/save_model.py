from keras.models import Model
from plant_seedlings.config import *

def save_model(model):
    """"
    Save the model to a file in the saved_models directory:
    input: model object. 
    Obs: model_dir, and file_name comes from config
    """

    if os.path.isdir(model_dir):
        if os.path.isfile(file_name):
            os.remove(file_name)
            model.save(file_name)
            print("model updated")
        else:
            model.save(file_name)
            print("model created")
    else:
        raise Exception("model_dir directory not found")
    
if __name__ == "__main__":
    from get_model import get_model
    model = get_model()
    save_model(model)