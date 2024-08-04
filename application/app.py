if __name__=="__main__":
	import streamlit as st
	import os
	import matplotlib.pyplot as plt
	import numpy as np
	from PIL import Image
	from keras.preprocessing.image import ImageDataGenerator
	from keras.applications.vgg16 import preprocess_input
	from keras.models import load_model
	
	#1. Create a streamlit title widget, this will be shown first
	st.title("Plant Seeds Identificator")
			
	upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
	c1, c2= st.columns(2)
	if upload is not None:
		# Open the image with PIL and reshape it to 224,224,3
		image= Image.open(upload)
		
		# Instantiate ImageDataGenerator to perform pre-processing on the loaded image
		image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
		image_resize = 224 # the target size of VGG
		
		# reshape it to 224,224,3 and display
		image_resized = image.resize((image_resize, image_resize))
		c1.image(image_resized)
		
		image_resized = np.array(image_resized) # as numpy array
		
		# use image_generator to transform the image_transformed
		x = np.expand_dims(image_resized, axis=0)
		img_transformed = image_generator.flow(x)
		
		c1.header('Input Image')
		#c1.write(y.shape)
		
		file_name = "saved_models/model.keras"
		if os.path.isfile(file_name):
			pass
		else:
			raise Exception("model not found!")
		
		#load the model
		model = load_model(file_name)
		
		# Evaluate on test_generator
		eval = model.predict(img_transformed)
		# transforms smt like [0.33,0.67] into [0,1]
		pred = np.argmax(eval,axis=-1) #(eval_vgg > 0.5).astype("int32")
			
		class_names = {0: 'Black-grass',
						1: 'Charlock',
						2: 'Cleavers',
						3: 'Common_Chickweed',
						4: 'Common_wheat',
						5: 'Fat_Hen',
						6: 'Loose_Silky-bent',
						7: 'Maize',
						8: 'Scentless_Mayweed',
						9: 'Shepherd_Purse',
						10: 'Small-flowered_Cranesbill',
						11: 'Sugar_beet'}
		# meaning: [0,1]: not_hot_dog
		#          [1,0]: hot_dog
			
		print("pred_vgg", int(pred),)
		ind = int(pred)
		prediction = class_names[ind]
			
		c2.header('Output')
		c2.subheader('The seed you have is :')
		c2.write(prediction)
