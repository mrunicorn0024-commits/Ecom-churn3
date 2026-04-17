from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# LOAD MODELS
# ==============================
rf_model = pickle.load(open("models/random_forest.pkl", "rb"))
svm_model = pickle.load(open("models/svm.pkl", "rb"))
lr_model = pickle.load(open("models/logistic_regression.pkl", "rb"))
gb_model = pickle.load(open("models/gradient_boosting.pkl", "rb"))

scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))


# ==============================
# HOME ROUTE
# ==============================
@app.route("/index")
def home():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/search")
def search():
    return render_template("search.html")

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


# ==============================
# PREDICTION API
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    model_name = data.get("model")
    features = data.get("features")

    # Convert to numpy array
    features = np.array(features).reshape(1, -1)

    # SCALE INPUT
    features = scaler.transform(features)

    # SELECT MODEL
    if model_name == "random_forest":
        model = rf_model
    elif model_name == "svm":
        model = svm_model
    elif model_name == "logistic_regression":
        model = lr_model
    elif model_name == "gradient_boosting":
        model = gb_model
    else:
        return jsonify({"error": "Invalid model selected"})

    # PREDICTION
    prediction = model.predict_proba(features)[0][1] * 100

    return jsonify({
        "churn_probability": round(prediction, 2)
    })


# ==============================
# SEARCH PREDICTION API
# ==============================
@app.route("/search_predict", methods=["POST"])
def search_predict():
    data = request.json

    features = data.get("features")
    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    prediction = rf_model.predict_proba(features)[0][1] * 100

    return jsonify({
        "churn_probability": round(prediction, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)