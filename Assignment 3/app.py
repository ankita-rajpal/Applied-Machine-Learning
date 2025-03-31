from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the model
model = joblib.load("models/model.pkl")

@app.route("/score", methods=["POST"])
def score_endpoint():
    """Flask endpoint to score a given text."""
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data["text"]
    prediction, propensity = score(text, model, 0.5)
    return jsonify({"prediction": prediction, "propensity": propensity})

@app.route("/")
def index():
    """Serve the index page."""
    return open("templates/index.html").read()

if __name__ == "__main__":
    app.run(debug=True)
