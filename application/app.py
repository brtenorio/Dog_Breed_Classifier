if __name__=="__main__":
	import streamlit as st
	import os
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	from PIL import Image
	from keras.preprocessing.image import ImageDataGenerator
	from keras.applications.resnet50 import preprocess_input
	from keras.models import load_model
	from activation_map import activation_map
	
	st.set_page_config(layout="centered")
	#1. Create a streamlit title widget, this will be shown first
	st.title("DOG BREED CLASSIFIER")
			
	upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
	c1, c2= st.columns(2)
	if upload is not None:
		# Open the image with PIL and reshape it to 224,224,3
		image= Image.open(upload)
		
		# Instantiate ImageDataGenerator to perform pre-processing on the loaded image
		image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
		image_resize = 224 # the target size 
		
		# reshape it to 224,224,3 and display
		image_resized = image.resize((image_resize, image_resize))

		c1.header('Input Image')

		# display the original image
		c1.image(image_resized)
		image_resized = np.array(image_resized) 
		
		# use image_generator to transform the image_transformed
		x = np.expand_dims(image_resized, axis=0)
		img_transformed = image_generator.flow(x)
		
		# Get the path to the model file and load it
		file_name = "saved_models/model.h5"
		file_name = os.path.normpath(file_name)
		if os.path.isfile(file_name):
			pass
		else:
			raise Exception("model not found!")
		model = load_model(file_name)
		
		# Predict the probabilities using the model
		prob = model.predict(img_transformed)[0] * 100
		# get the index of the classes with the maximum score
		top_classes = np.argsort(-prob)[:5]
		threshold = 10 # threshold % for displaying the class name
		
		# Read external file class_names.txt as dictionary class_names
		class_names = {}
		class_names_file = os.path.join(os.path.dirname(__file__) , 'class_names.txt')

		# Read the class names from the class_names.txt file
		with open(class_names_file, 'r') as file:
			for line in file:
				key, value = line.strip().split(':')
				class_names[int(key)] = value.strip()

		# Get the predicted class name and probability in the class_names dictionary
		prediction = {class_names[i].replace("_", " "):str(round(prob[i],1))+"%" for i in top_classes if prob[i] > threshold} 
		# convert the dictionary to a dataframe
		prediction_df = pd.DataFrame(list(prediction.items()), columns=['Class', 'Probability']).set_index('Class')
		c2.subheader('Prediction')
		c2.write(prediction_df)

		# Get class activation map
		cam = activation_map(model, image)
		cam = cam
			
		c2.header('Output')
		c2.subheader('The dogs breed is : \n' + str(prediction_df.index[0]))
		
		# Display the class activation map
		c2.image(cam, caption='Class Activation Map', use_column_width="auto")
