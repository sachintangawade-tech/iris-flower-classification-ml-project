"""
Simple helper to load the saved model and predict a species for a given feature vector.
"""
import joblib
import numpy as np

MODEL_PATH = 'iris_best_model.joblib'


def predict(features, model_path=MODEL_PATH):
    """
    Predict species name from features array-like [sepal length, sepal width, petal length, petal width].
    Returns the predicted species name as a string.
    """
    # Load model
    model = joblib.load(model_path)
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)
    return pred[0]


if __name__ == '__main__':
    # Demo sample
    sample = [5.1, 3.5, 1.4, 0.2]
    try:
        pred = predict(sample)
        print('Predicted species for sample', sample, '->', pred)
    except FileNotFoundError:
        print('Model file not found. Please run the notebook to train and save the model first.')
