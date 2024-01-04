import numpy as np
import pickle
import streamlit as st
import base64

# with open("group3_trained_logistic.sav", "rb") as file:
#     trained_model = pickle.load(file)

model_base64 = """
gASVpwYAAAAAAACMHnNrbGVhcm4ubGluZWFyX21vZGVsLl9sb2dpc3RpY5SMEkxvZ2lzdGljUmVncmVzc2lvbpSTlCmBlH2UKIwHcGVuYWx0eZSMAmwxlIwEZHVhbJSJjAN0b2yURz8aNuLrHEMtjAFDlEc/hHrhR64Ue4wNZml0X2ludGVyY2VwdJSIjBFpbnRlcmNlcHRfc2NhbGluZ5RLAYwMY2xhc3Nfd2VpZ2h0lE6MDHJhbmRvbV9zdGF0ZZROjAZzb2x2ZXKUjAlsaWJsaW5lYXKUjAhtYXhfaXRlcpRN9AGMC211bHRpX2NsYXNzlIwEYXV0b5SMB3ZlcmJvc2WUSwCMCndhcm1fc3RhcnSUiYwGbl9qb2JzlE6MCGwxX3JhdGlvlE6MEWZlYXR1cmVfbmFtZXNfaW5flIwVbnVtcHkuY29yZS5tdWx0aWFycmF5lIwMX3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RLAIWUQwFilIeUUpQoSwFLFYWUaBuMBWR0eXBllJOUjAJPOJSJiIeUUpQoSwOMAXyUTk5OSv////9K/////0s/dJRiiV2UKIwRSGlnaEJsb29kUHJlc3N1cmWUjA5IaWdoQ2hvbGVzdHJvbJSMHEhhZENob2xlc3Ryb2xDaGVja2VkSW41WWVhcnOUjA1Cb2R5TWFzc0luZGV4lIwMU21va2VyUGVyc29ulIwJSGFkU3Ryb2tllIwZSGFkSGVhcnREaXNlYXNlT3JBdHRhY2tlZJSME0hhZFBoeXNpY2FsQWN0aXZpdHmUjAlFYXRGcnVpdHOUjA1FYXRWZWdldGFibGVzlIwTSGVhdnlBbGNvaG9sRHJpbmtlcpSMEkhhdmVIZWFsdGhDb3ZlcmFnZZSMGU5vdE1ldERvY3RvckJlY2F1c2VPZkNvc3SUjA1HZW5lcmFsSGVhbHRolIwUTWVudGFsSGVhbHRoSW4zMERheXOUjBZQaHlzaWNhbEhlYWx0aEluMzBEYXlzlIwPRGlmZmljdWx0VG9XYWxrlIwDU2V4lIwDQWdllIwJRWR1Y2F0aW9ulIwGSW5jb21llGV0lGKMDm5fZmVhdHVyZXNfaW5flEsVjAhjbGFzc2VzX5RoGmgdSwCFlGgfh5RSlChLAUsDhZRoJIwCaTiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGAAAAAAAAAAAAQAAAAAAAAACAAAAAAAAAJR0lGKMBWNvZWZflGgaaB1LAIWUaB+HlFKUKEsBSwNLFYaUaCSMAmY4lImIh5RSlChLA2hKTk5OSv////9K/////0sAdJRiiEL4AQAAcn7Ne2xo5b+A8cHWIizFPxYobzbQ++Y/9BylYsli4r+zVhUVZ3TYP/eeHK0Gp+E/ov6Z6z/Y2r8AAAAAAAAAACP5cinsLtc/iM2kMb0+47947qPYs4+lP+Mdjxw+AeM/VThxwMn7mz8AAAAAAAAAANtO9R2En5q/ofD85H8npL8AAAAAAAAAABfXjr8QWa4/pclk/f7Mx78AAAAAAAAAADQvW9FRlso/ewVCAQ8slD8AAAAAAAAAAJbAJBCGYJ6/ORekvLuyoD8AAAAAAAAAAD361mcfA6W/idtbCX9qmj8AAAAAAAAAANmLuNpPd5q/R0VMO2HF4j8AAAAAAAAAAG9ijifaKOW/AAAAAAAAAADNOSxi0BmbvwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAh2rAlr5n3L/bsB01xgqjPxEuXudaj90/2snUfdAwwD8AAAAAAAAAADt5jb5LKsW/aBDWm6PPmD8AAAAAAAAAANXT30nSTJa/co70Xg8tur8AAAAAAAAAAHYJQrpmnL0/aPCaIcDgyL8AAAAAAAAAAPGYY4on98k/wXBeCiS0vL8WLs0pXaWnP7yZO4Bdjrs/cjlOzt4VrD/x0nVE+crIv9pMHOnzBai/noNiiuxGqj9VLPSb/lO0v4DAJousz6e/lHSUYowKaW50ZXJjZXB0X5RoGmgdSwCFlGgfh5RSlChLAUsDhZRoVYlDGO1Rr0whahlAwNcQvpP6CsC6lDR/m/0ZwJR0lGKMB25faXRlcl+UaBpoHUsAhZRoH4eUUpQoSwFLA4WUaCSMAmk0lImIh5RSlChLA2hKTk5OSv////9K/////0sAdJRiiUMMHQAAACUAAAAmAAAAlHSUYowQX3NrbGVhcm5fdmVyc2lvbpSMBTEuMi4ylHViLg==
"""

# Decode the Base64 string
model_bytes = base64.b64decode(model_base64)

# Deserialize the model
trained_model = pickle.loads(model_bytes)

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