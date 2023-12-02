import unittest
import app  # Your Flask app
import json
import numpy as np
import random
class APITestCase(unittest.TestCase):

    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()

    def test_predict_svm(self):
        # Replace with appropriate test data and expected result
        response = self.client.post('/predict/svm', json={'features': random.sample(range(0,255),64)})
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', json.loads(response.data))

    def test_predict_tree(self):
      
        response = self.client.post('/predict/tree', json={'features': random.sample(range(0,255),64) })
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', json.loads(response.data))

    def test_predict_lr(self):
    
        response = self.client.post('/predict/lr', json={'features': random.sample(range(0,255),64) })
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', json.loads(response.data))

if __name__ == '__main__':
    unittest.main()
