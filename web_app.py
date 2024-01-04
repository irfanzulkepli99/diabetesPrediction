import numpy as np
import pickle
import streamlit as st

with open("group3_trained_logistic.sav", "rb") as file:
    trained_model = pickle.load(file)

def predict_diabetes(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = trained_model.predict(input_data_reshaped)

    print(prediction)

    if prediction[0] == 0:
        return "No diabetes predicted."
    elif prediction[0] == 1:
        return "Pre-Diabetec predicted."
    elif prediction[0] == 2:
        return "Diabetes predicted."
    else:
        return "error"
    
def main():
    st.title("Diabetes Prediction Web App")

    BloodPressure = 1 if st.selectbox("High blood pressure ?", ["Yes", "No"]) == "Yes" else 0
    Cholestrol = 1 if st.selectbox("High cholestrol ?", ["Yes", "No"]) == "Yes" else 0
    CholestrolChecked = 1 if st.selectbox("Have checked cholestrol before ?", ["Yes", "No"]) == "Yes" else 0

    BMISelection = st.selectbox("Your BMI ?", ['<10', '10-19', '20-29', '30-39', '>= 40'])
    if BMISelection == '<10':
        BMI = 0
    elif BMISelection == '10-19':
        BMI = 2
    elif BMISelection == '20-29':
        BMI = 3
    elif BMISelection == '30-39':
        BMI = 4
    elif BMISelection == '>= 40':
        BMI = 5

    Smoker = 1 if st.selectbox("Are you a smoker ?", ["Yes", "No"]) == "Yes" else 0
    Stroke = 1 if st.selectbox("Have been stroked before ?", ["Yes", "No"]) == "Yes" else 0
    HeartDisease = 1 if st.selectbox("Having Heart Disease ?", ["Yes", "No"]) == "Yes" else 0
    PhysicalActivity = 1 if st.selectbox("Are you active physically ?", ["Yes", "No"]) == "Yes" else 0
    EatFruits = 1 if st.selectbox("Love to eat fruits ?", ["Yes", "No"]) == "Yes" else 0
    EatVegetables = 1 if st.selectbox("Love to eat vegetables ?", ["Yes", "No"]) == "Yes" else 0
    Alchoholic = 1 if st.selectbox("Are you a heavy drinkers ?", ["Yes", "No"]) == "Yes" else 0
    HealthcareCoverage = 1 if st.selectbox("Do you have healthcare coverage ?", ["Yes", "No"]) == "Yes" else 0
    NotMetDoctorBecauseOfCost = 1 if st.selectbox("Have you not met doctor before because of cost ?", ["Yes", "No"]) == "Yes" else 0

    GeneralHealthSelection = st.selectbox("General Health Scale ?", ['excellent', 'very good', 'good', 'fair', 'poor'])
    if GeneralHealthSelection == 'excellent':
        GeneralHealth = 1
    elif GeneralHealthSelection == 'very good':
        GeneralHealth = 2
    elif GeneralHealthSelection == 'good':
        GeneralHealth = 3
    elif GeneralHealthSelection == 'fair':
        GeneralHealth = 4
    elif GeneralHealthSelection == 'poor':
        GeneralHealth = 5

    MentalHealth30 = 1 if st.selectbox("Have you faced mental depression in 30 days ?", ['Less than one day', 'More than one day']) == 'Less than one day' else 2
    
    PhysicalHealth30 = 1 if st.selectbox("Have you faced physical health not good in 30 days ?", ['Less than one day', 'More than one day']) == 'Less than one day' else 2

    DifficultToWalk = 1 if st.selectbox("Facing difficulties of walking or climbing stairs ?", ["Yes", "No"]) == "Yes" else 0

    Sex = 1 if st.selectbox("Your gender ?", ["Male", "Female"]) == "Male" else 0

    age_categories = {
        "below than 18 years old": 1,
        "18-24 years old": 1,
        "25-29 years old": 2,
        "30-34 years old": 3,
        "35-39 years old": 4,
        "40-44 years old": 5,
        "45-49 years old": 6,
        "50-54 years old": 7,
        "55-59 years old": 8,
        "60-64 years old": 9,
        "65-69 years old": 10,
        "70-74 years old": 11,
        "75-79 years old": 12,
        "80 years old or older": 13
    }

    selected_age_range = st.selectbox("Select your age range:", list(age_categories.keys()))

    Age = age_categories[selected_age_range]

    education_levels = {
        "Never attended school or only kindergarten": 1,
        "Grades 1 through 8 (Elementary)": 2,
        "Grades 9 through 11 (Some high school)": 3,
        "Grade 12 or GED (High school graduate)": 4,
        "College 1 year to 3 years (Some college or technical school)": 5,
        "College 4 years or more (College graduate)": 6
    }

    selected_education_level = st.selectbox("Select your education level:", list(education_levels.keys()))

    Education = education_levels[selected_education_level]

    income_levels = {
        "Less than $10,000": 1,
        "Between $10,000 and $14,999": 2,
        "Between $15,000 and $19,999": 3,
        "Between $20,000 and $24,999": 4,
        "Less than $35,000": 5,
        "Between $35,000 and $49,999": 6,
        "Between $50,000 and $74,999": 7,
        "$75,000 or more": 8
    }

    selected_income_level = st.selectbox("Select your income level annually:", list(income_levels.keys()))

    Income = income_levels[selected_income_level]

    diagnosis = ''

    if st.button('Predict Diabetes Result'):
        diagnosis = predict_diabetes([BloodPressure,Cholestrol,CholestrolChecked,BMI,Smoker,Stroke,HeartDisease,PhysicalActivity,EatFruits,EatVegetables,Alchoholic,
                                      HealthcareCoverage,NotMetDoctorBecauseOfCost,GeneralHealth,MentalHealth30,PhysicalHealth30,DifficultToWalk,Sex,Age,Education,Income])
        
    st.success(diagnosis)


if __name__ == "__main__":
    main()