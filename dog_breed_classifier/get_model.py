from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import VGG16
from dog_breed_classifier.config import *

def get_model():

	base_model = VGG16(weights='imagenet', include_top=False) #imports the VGG16 model and discards the last layer.
	x = base_model.output 
	x = GlobalAveragePooling2D()(x) 

	x = Dropout(0.3)(x) 
	x = Dense(1024,activation='relu')(x) 
	x = Dropout(0.3)(x)
	x = Dense(512,activation='relu')(x) 

	preds = Dense(num_classes, activation='softmax')(x) #final layer with softmax activation

	model = Model(inputs=base_model.input, outputs=preds)

	#Freeze layers from VGG16 backbone (not to be trained)
	for layer in base_model.layers:
		layer.trainable=False
	
	model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
	
	return model

if __name__ == "__main__":
    pass