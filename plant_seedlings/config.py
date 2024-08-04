import os

# set a random state number
rs = 42

# set the path for the data base containing the images
file_path = "/Users/brncat/Downloads/AltaVerde/GitHub/seeds_db/"

# Define paths
main_dir = os.path.join(file_path,'main')
train_dir = os.path.join(file_path,'train')
val_dir = os.path.join(file_path,'valid')
test_dir = os.path.join(file_path,'test')

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

num_classes = 12

image_resize = 224

batch_size_training = 16
batch_size_validation = 16

num_epochs = 8

# set parameters to save the trained model
import saved_models
# The saved_models directory can be found after importing the module with the list [saved_models.__path__]
model_dir = saved_models.__path__[0]
model_name = "model.keras"
file_name = os.path.join(model_dir, model_name)
#check existence of model_dir
if os.path.isdir(model_dir):
    pass
else:
    raise Exception("saved_models directory not found!")