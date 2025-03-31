import unittest
import time
import requests
from score import score
import joblib
import subprocess

class TestFlaskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Launch the Flask app before tests."""
        cls.process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(3)  # Wait for the server to start
    
    @classmethod
    def tearDownClass(cls):
        """Close the Flask app after tests and release resources."""
        cls.process.terminate()
        cls.process.wait()
        cls.process.stdout.close()
        cls.process.stderr.close()
    
    def test_flask(self):
        """Test the Flask /score endpoint."""
        url = "http://127.0.0.1:5000/score"
        test_data = {"text": "Hello, win a free prize now!"}
        response = requests.post(url, json=test_data)
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction", json_data)
        self.assertIn("propensity", json_data)
        self.assertIsInstance(json_data["prediction"], bool)
        self.assertIsInstance(json_data["propensity"], float)
        self.assertGreaterEqual(json_data["propensity"], 0.0)
        self.assertLessEqual(json_data["propensity"], 1.0)

class TestScoreFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the trained model before testing."""
        cls.model = joblib.load("models/model.pkl")
    
    def test_score_function(self):
        """Smoke test: Ensure the function runs without crashing."""
        text = "Win a free iPhone now!"
        prediction, propensity = score(text, self.model, 0.5)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)
    
    def test_output_format(self):
        """Check if prediction is boolean and propensity is float."""
        text = "Win a lottery now!"
        prediction, propensity = score(text, self.model, 0.5)
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)
    
    def test_propensity_range(self):
        """Ensure propensity is between 0 and 1."""
        text = "Limited time offer!"
        _, propensity = score(text, self.model, 0.5)
        self.assertGreaterEqual(propensity, 0.0)
        self.assertLessEqual(propensity, 1.0)
    
    def test_threshold_zero(self):
        """If threshold is 0, prediction should always be 1."""
        text = "Congratulations! You've won!"
        prediction, _ = score(text, self.model, 0.0)
        self.assertTrue(prediction)
    
    def test_threshold_one(self):
        """If threshold is 1, prediction should always be 0."""
        text = "Hello, how are you?"
        prediction, _ = score(text, self.model, 1.0)
        self.assertFalse(prediction)
    
    def test_obvious_spam(self):
        """Test with an obvious spam message."""
        text = "1st wk FREE! Gr8 tones str8 2 u each wk. Txt NOKIA ON to 8007 for Classic Nokia tones or HIT ON to 8007 for Polys"
        prediction, _ = score(text, self.model, 0.5)
        self.assertTrue(prediction)
    
    def test_obvious_non_spam(self):
        """Test with an obvious non-spam message."""
        text = "Hi, I will meet you at 5 PM."
        prediction, _ = score(text, self.model, 0.5)
        self.assertFalse(prediction)

if __name__ == "__main__":
    unittest.main()
