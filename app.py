from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

data = pd.read_csv("student_data.csv")
data["StudyEfficiency"] = data["Assignments"] / data["StudyHours"]

X = data.drop("FinalScore", axis=1)
y = data["FinalScore"]

model = RandomForestRegressor()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        study = float(request.form["study"])
        assign = float(request.form["assign"])
        internal = float(request.form["internal"])
        sleep = float(request.form["sleep"])
        social = float(request.form["social"])

        efficiency = assign / study

        input_df = pd.DataFrame([[
            attendance, study, assign,
            internal, sleep, social, efficiency
        ]], columns=X.columns)

        prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)