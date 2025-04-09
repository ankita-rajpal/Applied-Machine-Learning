import unittest
import os
import time
import requests
import signal
import subprocess
from score import score
import joblib

# class TestFlaskIntegration(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         """Launch the Flask app before tests."""
#         cls.process = subprocess.Popen(
#             ["python", "app.py"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )
#         time.sleep(3)  # Wait for the server to start
    
#     @classmethod
#     def tearDownClass(cls):
#         """Close the Flask app after tests and release resources."""
#         cls.process.terminate()
#         cls.process.wait()
#         cls.process.stdout.close()
#         cls.process.stderr.close()
    
#     def test_flask(self):
#         """Test the Flask /score endpoint."""
#         url = "http://127.0.0.1:5001/score"
#         test_data = {"text": "Hello, win a free prize now!"}
#         response = requests.post(url, json=test_data)
        
#         self.assertEqual(response.status_code, 200)
#         json_data = response.json()
#         self.assertIn("prediction", json_data)
#         self.assertIn("propensity", json_data)
#         self.assertIsInstance(json_data["prediction"], bool)
#         self.assertIsInstance(json_data["propensity"], float)
#         self.assertGreaterEqual(json_data["propensity"], 0.0)
#         self.assertLessEqual(json_data["propensity"], 1.0)

# class TestScoreFunction(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         """Load the trained model before testing and ensure the file is properly closed."""
#         with open("models/model.pkl", "rb") as model_file:
#             cls.model = joblib.load(model_file)
    
#     def test_score_function(self):
#         """Smoke test: Ensure the function runs without crashing."""
#         text = "Win a free iPhone now!"
#         prediction, propensity = score(text, self.model, 0.5)
#         self.assertIsNotNone(prediction)
#         self.assertIsNotNone(propensity)
    
#     def test_output_format(self):
#         """Check if prediction is boolean and propensity is float."""
#         text = "Win a lottery now!"
#         prediction, propensity = score(text, self.model, 0.5)
#         self.assertIsInstance(prediction, bool)
#         self.assertIsInstance(propensity, float)
    
#     def test_propensity_range(self):
#         """Ensure propensity is between 0 and 1."""
#         text = "Limited time offer!"
#         _, propensity = score(text, self.model, 0.5)
#         self.assertGreaterEqual(propensity, 0.0)
#         self.assertLessEqual(propensity, 1.0)
    
#     def test_threshold_zero(self):
#         """If threshold is 0, prediction should always be 1."""
#         text = "Congratulations! You've won!"
#         prediction, _ = score(text, self.model, 0.0)
#         self.assertTrue(prediction)
    
#     def test_threshold_one(self):
#         """If threshold is 1, prediction should always be 0."""
#         text = "Hello, how are you?"
#         prediction, _ = score(text, self.model, 1.0)
#         self.assertFalse(prediction)
    
#     def test_obvious_spam(self):
#         """Test with an obvious spam message."""
#         text = "Claim your free gift now! Click here!"
#         prediction, _ = score(text, self.model, 0.5)
#         self.assertTrue(prediction)
    
#     def test_obvious_non_spam(self):
#         """Test with an obvious non-spam message."""
#         text = "Hi, I will meet you at 5 PM."
#         prediction, _ = score(text, self.model, 0.5)
#         self.assertFalse(prediction)

class TestDockerizedApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.system("docker build -t flask-score-app .")
        cls.container_id = os.popen("docker run -d -p 5000:5000 flask-score-app").read().strip()
        time.sleep(5)  # Allow time for the container to boot

    @classmethod
    def tearDownClass(cls):
        os.system(f"docker stop {cls.container_id}")
        os.system(f"docker rm {cls.container_id}")

    def test_docker(self):
        url = "http://127.0.0.1:5000/score"
        test_data = {"text": "hi! how are you?"}
        response = requests.post(url, json=test_data)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction", json_data)
        self.assertIn("propensity", json_data)
        self.assertIsInstance(json_data["prediction"], bool)
        self.assertIsInstance(json_data["propensity"], float)

if __name__ == "__main__":
    unittest.main()
