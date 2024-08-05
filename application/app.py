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
	st.title("DOG BREED CLASSIFIER")
			
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
		
		file_name = "saved_models/model.h5"
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
			
		class_names = {0: 'Afghan_hound',
				1: 'African_hunting_dog',
				2: 'Airedale',
				3: 'American_Staffordshire_terrier',
				4: 'Appenzeller',
				5: 'Australian_terrier',
				6: 'Bedlington_terrier',
				7: 'Bernese_mountain_dog',
				8: 'Blenheim_spaniel',
				9: 'Border_collie',
				10: 'Border_terrier',
				11: 'Boston_bull',
				12: 'Bouvier_des_Flandres',
				13: 'Brabancon_griffon',
				14: 'Brittany_spaniel',
				15: 'Cardigan',
				16: 'Chesapeake_Bay_retriever',
				17: 'Chihuahua',
				18: 'Dandie_Dinmont',
				19: 'Doberman',
				20: 'English_foxhound',
				21: 'English_setter',
				22: 'English_springer',
				23: 'EntleBucher',
				24: 'Eskimo_dog',
				25: 'French_bulldog',
				26: 'German_shepherd',
				27: 'German_short-haired_pointer',
				28: 'Gordon_setter',
				29: 'Great_Dane',
				30: 'Great_Pyrenees',
				31: 'Greater_Swiss_Mountain_dog',
				32: 'Ibizan_hound',
				33: 'Irish_setter',
				34: 'Irish_terrier',
				35: 'Irish_water_spaniel',
				36: 'Irish_wolfhound',
				37: 'Italian_greyhound',
				38: 'Japanese_spaniel',
				39: 'Kerry_blue_terrier',
				40: 'Labrador_retriever',
				41: 'Lakeland_terrier',
				42: 'Leonberg',
				43: 'Lhasa',
				44: 'Maltese_dog',
				45: 'Mexican_hairless',
				46: 'Newfoundland',
				47: 'Norfolk_terrier',
				48: 'Norwegian_elkhound',
				49: 'Norwich_terrier',
				50: 'Old_English_sheepdog',
				51: 'Pekinese',
				52: 'Pembroke',
				53: 'Pomeranian',
				54: 'Rhodesian_ridgeback',
				55: 'Rottweiler',
				56: 'Saint_Bernard',
				57: 'Saluki',
				58: 'Samoyed',
				59: 'Scotch_terrier',
				60: 'Scottish_deerhound',
				61: 'Sealyham_terrier',
				62: 'Shetland_sheepdog',
				63: 'Shih-Tzu',
				64: 'Siberian_husky',
				65: 'Staffordshire_bullterrier',
				66: 'Sussex_spaniel',
				67: 'Tibetan_mastiff',
				68: 'Tibetan_terrier',
				69: 'Walker_hound',
				70: 'Weimaraner',
				71: 'Welsh_springer_spaniel',
				72: 'West_Highland_white_terrier',
				73: 'Yorkshire_terrier',
				74: 'affenpinscher',
				75: 'basenji',
				76: 'basset',
				77: 'beagle',
				78: 'black-and-tan_coonhound',
				79: 'bloodhound',
				80: 'bluetick',
				81: 'borzoi',
				82: 'boxer',
				83: 'briard',
				84: 'bull_mastiff',
				85: 'cairn',
				86: 'chow',
				87: 'clumber',
				88: 'cocker_spaniel',
				89: 'collie',
				90: 'curly-coated_retriever',
				91: 'dhole',
				92: 'dingo',
				93: 'flat-coated_retriever',
				94: 'giant_schnauzer',
				95: 'golden_retriever',
				96: 'groenendael',
				97: 'keeshond',
				98: 'kelpie',
				99: 'komondor',
				100: 'kuvasz',
				101: 'malamute',
				102: 'malinois',
				103: 'miniature_pinscher',
				104: 'miniature_poodle',
				105: 'miniature_schnauzer',
				106: 'otterhound',
				107: 'papillon',
				108: 'pug',
				109: 'redbone',
				110: 'schipperke',
				111: 'silky_terrier',
				112: 'soft-coated_wheaten_terrier',
				113: 'standard_poodle',
				114: 'standard_schnauzer',
				115: 'toy_poodle',
				116: 'toy_terrier',
				117: 'vizsla',
				118: 'whippet',
				119: 'wire-haired_fox_terrier'}
			
		print("pred_vgg", int(pred),)
		ind = int(pred)
		prediction = class_names[ind]
			
		c2.header('Output')
		c2.subheader('The seed you have is :')
		c2.write(prediction)
