import os

"""This module contains the configuration parameters for the dog breed classifier model"""

# set a random state number
rs = 42

# set the path for the data base containing the images
file_path = "/Users/brncat/Downloads/AltaVerde/GitHub/Dog_Breed_Dataset"

#check existence of the dataset path: file_path
if os.path.isdir(file_path):
    print("data set found!")
else:
    raise Exception("data set directory not found!")

# number of classes
num_classes = 120

# # Define paths
main_dir = os.path.join(file_path,'main')
train_dir = os.path.join(file_path,'train')
val_dir = os.path.join(file_path,'valid')
test_dir = os.path.join(file_path,'test')

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

image_resize = 224

batch_size_training = 56
batch_size_validation = 56

num_epochs = 16 

# Define the path for the model
file_name = "saved_models/model.h5"
