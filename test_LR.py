import unittest
import joblib
from sklearn.linear_model import LogisticRegression

class TestLogisticRegressionModel(unittest.TestCase):

    def test_model_type(self):
        # Load the model and check if it's a Logistic Regression model
        model_path = '/mnt/d/Mtech_Practice/MLOps/Lectures/hand_written_digits_recognition/m22aie244_lr_lbfgs.joblib'
        model = joblib.load(model_path)
        self.assertIsInstance(model, LogisticRegression, "Loaded model is not a Logistic Regression model")

    def test_solver_name(self):
        # Check if the solver name in the model matches the expected solver name
        solver_name = "liblinear"
        model_path = '/mnt/d/Mtech_Practice/MLOps/Lectures/hand_written_digits_recognition/m22aie244_lr_liblinear.joblib'
        model = joblib.load(model_path)
        self.assertEqual(model.get_params()['solver'], solver_name, "Solver name in the model does not match '{}'".format(solver_name))

if __name__ == '__main__':
    unittest.main()
