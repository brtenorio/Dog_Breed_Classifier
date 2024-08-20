from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import ResNet50
from dog_breed_classifier.config import *

def get_model():
	"""
	Load the ResNet50 model with ImageNet weights and add a few layers on top of it.
	"""

	base_model = ResNet50(input_shape=[image_resize, image_resize, 3],
                          weights='imagenet', include_top=False)

	#Freeze layers from ResNet backbone (not to be trained)
	for layer in base_model.layers:
		layer.trainable=False

	x = base_model.output
	x = GlobalAveragePooling2D(name="GlobalAveragePooling2D")(x)
	x = Dropout(0.3, name="Dopout")(x)
	preds = Dense(num_classes, activation='softmax', name='predictions')(x)

	model = Model(inputs=base_model.input, outputs=preds)

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Make the last convolutional layer trainable for fine tuning
	model.get_layer('conv5_block3_3_conv').trainable = True
	model.get_layer('conv5_block3_3_bn').trainable = True

	return model

if __name__ == "__main__":
    pass
