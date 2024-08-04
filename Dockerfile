# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Update pip
RUN pip install --upgrade pip

# Install HDF5 and its dependencies using apt
RUN apt-get update && apt-get install -y pkg-config gcc libhdf5-dev

# Copy the rest of the application code to the working directory
COPY . /app
WORKDIR /app

# Install the app dependencies 
RUN python3 -m pip install -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "application/app.py", "--server.port", "8080"]

