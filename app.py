import pickle
from flask import Flask, request, app, url_for, render_template
import pandas as pd
import numpy as np
import sklearn

app = Flask(__name__)

model = pickle.load(open("rf_model.pkl", 'rb'))
#df = pickle.load(open("CR_data.pkl", "rb"))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    Marital_status = request.form["Marital_status"]
    Gender= request.form['Gender']
    Number_of_dependents= request.form['Number_of_dependents']
    Housing =request.form['Housing']
    Years_at_current_residence = request.form['Years_at_current_residence']
    Employment_status = request.form['Employment_status']
    Foreign_worker=request.form['Foreign_worker']
    Savings_account_balance=request.form['Savings_account_balance']
    Has_been_employed_for_at_least=request.form['Has_been_employed_for_at_least']
    Purpose=request.form['Purpose']
    EMI_rate_in_percentage_of_disposable_income=request.form['EMI_rate_in_percentage_of_disposable_income']
    Property=request.form['Property']
    Has_coapplicant=request.form['Has_coapplicant']
    Has_guarantor=request.form['Has_guarantor']
    Number_of_existing_loans_at_this_bank=request.form['Number_of_existing_loans_at_this_bank']
    Loan_history=request.form['Loan_history']


    new_data={}

    new_data['Marital_status']= Marital_status
    new_data['Gender'] = Gender
    new_data['Number_of_dependents'] = Number_of_dependents
    new_data['Housing'] =Housing
    new_data['Years_at_current_residence'] = Years_at_current_residence
    new_data['Employment_status'] = Employment_status
    new_data['Foreign_worker'] = Foreign_worker
    new_data['Savings_account_balance'] = Savings_account_balance
    new_data['Has_been_employed_for_at_least'] = Has_been_employed_for_at_least
    new_data['Purpose'] = Purpose
    new_data['EMI_rate_in_percentage_of_disposable_income'] = EMI_rate_in_percentage_of_disposable_income
    new_data['Property'] = Property
    new_data['Has_coapplicant'] = Has_coapplicant
    new_data['Has_guarantor'] = Has_guarantor
    new_data['Number_of_existing_loans_at_this_bank'] = Number_of_existing_loans_at_this_bank
    new_data['Loan_history'] = Loan_history

    input_values = [float(x) for x in new_data.values()]
    arr_val = np.array(input_values).reshape(1, -1)
    prediction = model.predict(arr_val)
    output = prediction[0]

    # Prediction Result :
    result = ""
    if output == 0:
        result = "Applicant has low credit risk i.e high chance of paying back the loan amount"
    else:
        result = "Applicant has high credit risk i.e low chance of paying back the loan amount"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
