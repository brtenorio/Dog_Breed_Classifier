from dog_breed_classifier.config import *

def train_model(model, train_generator, validation_generator):
        """
        This function trains the model using the training and validation generators.
        """
        steps_per_epoch_training = len(train_generator)
        steps_per_epoch_validation = len(validation_generator)
        model.fit(train_generator,
            steps_per_epoch=steps_per_epoch_training,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch_validation,
            verbose="auto"
            )
        return model