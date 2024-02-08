from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # For classification tasks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # For classification tasks
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("Loan_default.csv")
Y = df["Default"]
X = df.loc[:, ["Age", "InterestRate", "Income", "LoanAmount", "MonthsEmployed", "MaritalStatus"]]
X = pd.get_dummies(X, columns=["MaritalStatus"], drop_first=True)
rus = RandomUnderSampler(random_state=42)
X_resampled, Y_resampled = rus.fit_resample(X, Y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            age = [int(request.form.get("age"))]
            month = [int(request.form.get("months"))]
            salary = [float(request.form.get("salary"))]
            loan = [float(request.form.get("loan"))]
            rate = [float(request.form.get("rate"))]
            marital = request.form.get("marital")

            # Dynamically create prediction dataframe based on marital status
            pred_df = pd.DataFrame({
                "Age": age,
                "InterestRate": rate,
                "Income": salary,
                "LoanAmount": loan,
                "MonthsEmployed": month,
                "MaritalStatus_Married": [1 if marital == "married" else 0],
                "MaritalStatus_Single": [1 if marital == "single" else 0]
            })

            result = rf_classifier.predict(pred_df)
            result_text = "High Risky." if result == 1 else "Low Risky"
            return render_template("index.html", result=result_text)
        else:
            return render_template("index.html", result="Waiting for input...")
    except Exception as e:
        return render_template("index.html", result=f"Please give valid input")

if __name__ == "__main__":
    app.run()
