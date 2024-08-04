import os
import shutil
from sklearn.model_selection import train_test_split
from plant_seedlings.config import *

def create_dir():
	if os.path.isdir(file_path):
		if os.path.isdir(train_dir):
			print("train dir exist")
		else:
			print("creating train valid test sets")
			# Iterate through each class
			for class_dir in os.listdir(main_dir):
				class_path = os.path.join(main_dir, class_dir)
				if os.path.isdir(class_path):
					# Create train, val, test directories for the class
					os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
					os.makedirs(os.path.join(val_dir, class_dir), exist_ok=True)
					os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)
			
					# Get all images in the class directory as a list of strings
					images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
			
					# Shuffle and split data
					train_imgs, test_imgs = train_test_split(images, test_size=(1-train_ratio), shuffle=True, 
													random_state=rs)
					val_imgs, test_imgs = train_test_split(test_imgs, test_size=(test_ratio/(val_ratio+test_ratio)), 
												shuffle=True,random_state=rs)
			
					# Copy files to corresponding directories
					for img in train_imgs:
						shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_dir, img))
					for img in val_imgs:
						shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_dir, img))
					for img in test_imgs:
						shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_dir, img))
	else:
		raise Exception("data set directory not found!")

if __name__ == "__main__":
	create_dir()
