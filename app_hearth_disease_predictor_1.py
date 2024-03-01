import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost


st.title('Predición de Enfermedades Cardiacas')

# Carga tu modelo XGBoost
model = joblib.load("C:\\Users\\mbrou\\OneDrive\\Documentos\\Master Big Data & Data Science\\TFM\\TFM Entrega\\modelo_XGB1.sav.sav")

def input_to_features():
    general_health = st.selectbox('Salud General', ['Excelente', 'Muy Buena', 'Buena', 'Regular', 'Mala'])
    checkup = st.selectbox('Checkup', ['Durante el último año', 'Durante los últimos 2 años', 'Durante los últimos 5 años', 'Hace más de 5 añoos', 'Nunca'])
    exercise = st.radio('Ejercicio', ['Si', 'No'])
    skin_cancer = st.radio('Cáncer de Piel', ['Si', 'No'])
    other_cancer = st.radio('Otros tipos de Cáncer', ['Si', 'No'])
    depression = st.radio('Depresión', ['Si', 'No'])
    diabetes = st.selectbox('Diabetes', ['No', 'Si', 'No, pre-diabetes', 'Si, pero solo durante el embarazo'])
    arthritis = st.radio('Artritis', ['Si', 'No'])
    sex = st.radio('Sexo', ['Masculino', 'Femenino'])
    age_category = st.selectbox('Categoría de Edad', ['Joven (18-24)', 'Adulto (25-39)', 'Adulto de Mediana Edad (40-54)', 'Adulto de Tercera Edad (55-64)', 'Anciano (65 o mayor)'])
    smoking_history = st.radio('Historial de Fumar', ['Sí', 'No'])
    BMI = st.selectbox('IMB', ['Peso Bajo', 'Peso Sano', 'Sobrepeso', 'Obesidad I', 'Obesidad II', 'Obesidad Grave'])
    alcohol_consumption = st.slider('Consumo de Alcohol en los últimos 30 días', 0, 30)
    fruit_consumption = st.slider('Consumo de Frutas en los últimos 30 días', 0, 30)
    green_vegetables_consumption = st.slider('Consumo de Vegetales Verdes en los últimos 30 días', 0, 30)
    friedpotato_consumption = st.slider('Consumo de Papas Fritas en los últimos 30 días', 0, 30)
    data = {
            'General_Health': general_health,
            'Checkup': checkup,
            'Exercise': exercise,
            'Skin_Cancer': skin_cancer,
            'Other_Cancer': other_cancer,
            'Depression': depression,
            'Diabetes': diabetes,
            'Arthritis': arthritis,
            'Sex': sex,
            'Age_Category': age_category,
            'Smoking_History': smoking_history,
            'Alcohol_Consumption': alcohol_consumption,
            'Fruit_Consumption': fruit_consumption,
            'Green_Vegetables_Consumption': green_vegetables_consumption,
            'FriedPotato_Consumption': friedpotato_consumption,
            'BMI_group': BMI}
    features = pd.DataFrame(data, index=[0])
    return features

dff = input_to_features()

general_health_map = {'Excelente': 4, 'Muy Buena': 3, 'Buena': 2, 'Regular': 1, 'Mala': 0}
checkup_map = {'Durante el último año': 0, 'Durante los últimos 2 años': 1, 'Durante los últimos 5 años': 2, 'Hace más de 5 años': 3, 'Nunca': 4}
exercise_map = {'Si': 1, 'No': 0}
skin_cancer_map = {'Si': 1, 'No': 0}
other_cancer_map = {'Si': 1, 'No': 0}
depression_map = {'Si': 1, 'No': 0}
diabetes_map = {'No': 0, 'Si': 1, 'No, pre-diabetes': 2, 'Si, pero solo durante el embarazo': 3}
arthritis_map = {'Si': 1, 'No': 0}
sex_map = {'Masculino': 1, 'Femenino': 0}
age_category_map = {'Joven (18-24)': 0, 'Adulto (25-39)': 1, 'Adulto de Mediana Edad (40-54)': 2, 'Adulto de Tercera Edad (55-64)': 3, 'Anciano (65 o mayor)': 4}
smoking_history_map = {'Si': 1, 'No': 0}
BMI_map = {'Peso Bajo': 0, 'Peso Sano': 1, 'Sobrepeso': 2, 'Obesidad I': 3, 'Obesidad II': 4, 'Obesidad Grave': 5}

dff['General_Health'] = dff['General_Health'].map(general_health_map)
dff['Checkup'] = dff['Checkup'].map(checkup_map)
dff['Exercise'] = dff['Exercise'].map(exercise_map)
dff['Skin_Cancer'] = dff['Skin_Cancer'].map(skin_cancer_map)
dff['Other_Cancer'] = dff['Other_Cancer'].map(other_cancer_map)
dff['Depression'] = dff['Depression'].map(depression_map)
dff['Diabetes'] = dff['Diabetes'].map(diabetes_map)
dff['Arthritis'] = dff['Arthritis'].map(arthritis_map)
dff['Sex'] = dff['Sex'].map(sex_map)
dff['Age_Category'] = dff['Age_Category'].map(age_category_map)
dff['Smoking_History'] = dff['Smoking_History'].map(smoking_history_map)
dff['BMI_group'] = dff['BMI_group'].map(BMI_map)


if st.button('Predict'):
    prediction = model.predict(dff)
    prediction = pd.DataFrame(prediction, columns=['prediction'])
    heart_disease_prediction = prediction['prediction'].iloc[0]
    
    if heart_disease_prediction == 1:
        st.error('El paciente puede padecer problemas cardíacos')
    else:
        st.success('El paciente no padece problemas cardíacos')