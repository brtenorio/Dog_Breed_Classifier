if __name__=="__main__":
    """
    Run this script if you want to retrain the model and persist a new model.keras file
    """
    from dog_breed_classifier.data_generator import data_generator
    from dog_breed_classifier.train_model import train_model
    from dog_breed_classifier.get_model import get_model
    from dog_breed_classifier.evaluate_model import evaluate_model
    from dog_breed_classifier.save_model import save_model

    model = get_model()
    train_generator, validation_generator, test_generator = data_generator()
    model = train_model(model,train_generator, validation_generator)
    save_model(model)
    eval = evaluate_model(test_generator)
    print('Test loss: ', eval[0])
    print('Test Accuracy: ', eval[1])