from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    client_id = float(request.form.get('Client_ID'))
    client_income = float(request.form.get('Client_Income'))
    car_owned = float(request.form.get('Car_Owned'))
    bike_owned = float(request.form.get('Bike_Owned'))
    active_loan = float(request.form.get('Active_Loan'))
    credit_amount = float(request.form.get('Credit_Amount'))
    loan_annuity =  float(request.form.get('Loan_Annuity'))
    client_income_type = float(request.form.get('client_income_type'))
    client_education = float(request.form.get('client_education'))
    client_marital_status = float(request.form.get('client_marital_status'))
    client_gender = float(request.form.get('client_gender'))
    loan_contract_type = request.form.get('loan_contract_type')
    client_occupation = float(request.form.get('client_occupation'))
    type_organization = float(request.form.get('type_organization'))
    house_own = float(request.form.get('House_Own'))
    child_count = float(request.form.get('child_count'))
    accompany_clint = float(request.form.get('Accompany_client'))
    client_housing = float(request.form.get('client_housing_type'))
    population_region_relative = float(request.form.get('population_region_relative'))
    age_days = float(request.form.get('Age_days'))
    employed_days= float(request.form.get('Employed_days'))
    registration_days= float(request.form.get('Registration_days'))
    id_day = float(request.form.get('Id_days'))
    own_house_age = float(request.form.get('Own_house_age'))
    mobile_tag = float(request.form.get('mobile_tag'))
    homephone_tag = float(request.form.get('Homephone_tag'))
    workphone_working = float(request.form.get('workphone_working'))
    social_circle_default = float(request.form.get('social_circle_default'))
    phone_change = float(request.form.get('phone_change'))
    credit_bureau = float(request.form.get('Credit_bureau'))
    client_family_memebers = float(request.form.get('client_family_members'))
    client_city_rating = float(request.form.get('client_city_rating'))
    application_process_days = float(request.form.get('application_process_days'))
    application_process_hour = float(request.form.get('application_process_hour'))
    client_permant_match_tag = float(request.form.get('client_permant_match_tag'))
    client_contact_work_tag = float(request.form.get('client_contact_work_tag'))
    score_source_1 = float(request.form.get('score_source_1'))
    score_source_2 =  float(request.form.get('score_source_2'))
    score_source_3 = float(request.form.get('score_source_3'))

    data = [
        client_id, client_income, car_owned,
        bike_owned, active_loan, credit_amount,
        loan_annuity, client_income_type,
        client_education,
        client_marital_status,
        client_gender, loan_contract_type,
        client_occupation,
        type_organization,house_own,child_count,accompany_clint,
        client_housing,population_region_relative,age_days,employed_days,
        registration_days,id_day,own_house_age,mobile_tag,homephone_tag,
        workphone_working,social_circle_default,phone_change,credit_bureau,
        client_family_memebers,client_city_rating,application_process_days,
        application_process_hour,client_permant_match_tag,client_contact_work_tag,
        score_source_1,score_source_2,score_source_3,
    ]
    

    # Perform prediction or processing here (not implemented in this example)
    data = [int(x) for x in data]
    final_features = [np.array(data)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 1)

    # Render the form again with submitted values and prediction result
    return render_template('index.html',
                           prediction_text="Result: {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)