from plant_seedlings.config import *
from plant_seedlings.data_generator import *

def train_model(model):
        model.fit(train_generator,
            steps_per_epoch=steps_per_epoch_training,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch_validation,
            verbose="auto"
            )
        return model