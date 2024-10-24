def save_model(model, filepath):
    """
    Save the trained model to a file.

    Parameters:
        model (sklearn model): Trained model to save.
        filepath (str): Path to save the model.
    """
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a trained model from a file.

    Parameters:
        filepath (str): Path to load the model from.

    Returns:
        sklearn model: Loaded model.
    """
    import joblib
    return joblib.load(filepath)