from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from plant_seedlings.config import *

#check existence of the dataset path: file_path
if os.path.isdir(file_path):
    print("data set found!")
else:
    raise Exception("data set directory not found!")

# Instantiate the image data generator
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical',
    seed=rs
    )

validation_generator = data_generator.flow_from_directory(
    val_dir,
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical',
    seed=rs
)

test_generator = data_generator.flow_from_directory(
    test_dir,
    target_size=(image_resize, image_resize),
    shuffle=True,
    batch_size = 1,
    class_mode='categorical',
    seed=rs
)

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)

# Optionally, the number of classes could be extracted automatically from data_generator
#num_classes = train_generator.num_classes

print(steps_per_epoch_training,steps_per_epoch_validation)