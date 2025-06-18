import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load("model.pkl")  # make sure model.pkl is created after training
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Iris example
    prediction = model.predict(sample_input)
    assert prediction.shape == (1,), "Prediction output is not correct"

if __name__ == "__main__":
    test_model_prediction()
    print("✅ Model test passed!")