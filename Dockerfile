# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Update pip
RUN pip install --upgrade pip

# Install HDF5 and its dependencies using apt
RUN apt-get update && apt-get install -y pkg-config gcc libhdf5-dev

# Isntall git large file system
RUN apt-get install git-lfs -y

RUN git clone https://github.com/brtenorio/Dog_Breed_Classifier.git

# Set the cloned repo the working directory inside the container
WORKDIR Dog_Breed_Classifier

# Install the app dependencies 
RUN python3 -m pip install -r requirements.txt

# Fetch large files
RUN git lfs fetch --all && git pull && git lfs pull

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "application/app.py", "--server.port", "8080"]
