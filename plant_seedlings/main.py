if __name__=="__main__":
    """
    Run this script if you want to retrain the model and persist a new model.keras file
    """
    from plant_seedlings.data_generator import * 
    from plant_seedlings.train_model import train_model
    from plant_seedlings.get_model import get_model
    from plant_seedlings.evaluate_model import evaluate_model
    from plant_seedlings.save_model import save_model

    model = get_model()
    model = train_model(model)
    save_model(model)
    eval = evaluate_model(test_generator)
    print('Test loss: ', eval[0])
    print('Test Accuracy: ', eval[1])