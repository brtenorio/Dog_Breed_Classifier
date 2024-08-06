# Dog Breed Classifier

![](classifier.jpg)

Dog_Breed_Classifier is a Convolutional Neural Network (CNN) application based on the VGG16 model for recognizing among 120 dog breeds. This project uses Streamlit for the web interface, Poetry for dependency management, and Docker to containerize the app.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [License](#license)

## Installation

To install and set up the project, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/brtenorio/Dog_Breed_Classifier
    cd dog_Breed_Classifier
    ```

2. **Install Poetry (if not already installed):**
    ```sh
    pip install poetry==1.8.3 
    ```

3. **Set up the virtual environment and install dependencies:**
    ```sh
    make all
    ```

## Usage

To run the application, use:

    make run

## Development

To retrain the model, if needed, use:

    make retrain-model

## Running Tests

To test the model, use:

    make test

## Docker

To containerize the application, use:

    make docker-build

followed by
    
    make docker-run

