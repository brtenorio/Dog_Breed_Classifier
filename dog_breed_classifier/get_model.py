from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import ResNet50
from dog_breed_classifier.config import *

def get_model():
	"""
	Load the ResNet50 model with ImageNet weights and add a few layers on top of it.
	"""

	base_model = ResNet50(input_shape=[image_resize, image_resize, 3],
							weights='imagenet', include_top=False) #imports the ResNet50 model and discards the last layer.
	x = base_model.output # (None, None, None, 512)
	x = GlobalAveragePooling2D()(x) # (None, 512)

	x = Dropout(0.3)(x) # (None, 512)
	x = Dense(1024,activation='relu')(x) # (None, 1024)
	x = Dropout(0.3)(x) # (None, 512)
	x = Dense(512,activation='relu')(x) # (None, 512)

	preds = Dense(num_classes,activation='softmax')(x) #final layer with softmax activation

	model = Model(inputs=base_model.input, outputs=preds)

	#Freeze layers from VGG16 backbone (not to be trained)
	for layer in base_model.layers:
		layer.trainable=False

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model

if __name__ == "__main__":
    pass
