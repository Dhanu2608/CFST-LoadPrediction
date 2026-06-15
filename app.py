from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("cfst_multi_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        D = float(data["Diameter"])
        H = float(data["Height"])
        t = float(data["Thickness"])
        F = float(data["Fibre"])
        T = float(data["Taper"])
        C = float(data["Concrete"])

        input_data = pd.DataFrame([{
            "Diameter": D,
            "Fibre": F,
            "Taper": T,
            "Thickness": t,
            "Concrete": C,
            "Height": H,
            "H_D": H / D,
            "D_t": D / t,
            "Slenderness": H / D,
            "Thickness_Ratio": t / D,
            "Fibre_Concrete": F * C
        }])

        prediction = model.predict(input_data)

        return jsonify({
            "Axial Load": float(prediction[0][0]),
            "Lateral Constant": float(prediction[0][1]),
            "Lateral Variable": float(prediction[0][2])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)