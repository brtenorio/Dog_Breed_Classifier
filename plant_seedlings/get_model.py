from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from plant_seedlings.config import *
#from data_generator import *

def get_model():
	input = Input(shape=(image_resize,image_resize,3,)) # (None, 224, 224, 3)
	
	conv1  = Conv2D(32,(2,2),activation='relu',padding="same")(input) 
	conv12 = Conv2D(32,(2,2),activation='relu',padding="same")(conv1) 
	pool1  = MaxPooling2D(pool_size=(2,2))(conv12) 
	
	conv2  = Conv2D(64,(3,3),activation='relu',padding="same")(pool1) 
	conv22 = Conv2D(64,(3,3),activation='relu',padding="same")(conv2) 
	pool2  = MaxPooling2D(pool_size=(2,2))(conv22) 
	
	conv3 = Conv2D(128,(3,3),activation='relu',padding="same")(pool2) 
	conv32 = Conv2D(128,(3,3),activation='relu',padding="same")(conv3) 
	pool3 = MaxPooling2D(pool_size=(2,2))(conv32)
	
	conv4 = Conv2D(256,(3,3),activation='relu')(pool3)
	pool4 = MaxPooling2D(pool_size=(3,3))(conv4)

	conv5 = Conv2D(256,(3,3),activation='relu')(pool4)
	pool5 = MaxPooling2D(pool_size=(3,3))(conv5) 
	
	flat = Flatten()(pool5) 
	
	drop = Dropout(0.3)(flat) 
	fully1 = Dense(512, activation='relu')(drop)  # (None, 512)
	drop1 = Dropout(0.3)(fully1) 
	fully2 = Dense(512, activation='relu')(drop1) # (None, 512)
	
	pred = Dense(num_classes, activation='softmax')(fully2) # (None, 12)
	
	model = Model(inputs=input, outputs=pred)
	
	model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
	
	return model

if __name__ == "__main__":
    pass