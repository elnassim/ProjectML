import streamlit as st
import requests

st.title("Prédiction de Revenu Annuel 🎯")

# Input form
with st.form("input_form"):
    age = st.number_input("Âge", min_value=18, max_value=70)
    area = st.selectbox("Zone", ["urbain", "rural"])
    gender = st.selectbox("Genre", ["homme", "femme"])
    education = st.selectbox("Éducation", ["sans_niveau", "fondamental", "secondaire", "supérieur"])
    years_experience = st.number_input("Années d'expérience", min_value=0, max_value=40)
    marital_status = st.selectbox("État matrimonial", ["célibataire", "marié", "divorcé", "veuf"])
    socio_group = st.selectbox("Catégorie socio-professionnelle", ["Groupe1", "Groupe2", "Groupe3", "Groupe4", "Groupe5", "Groupe6"])
    
    submitted = st.form_submit_button("Prédire")
    
    if submitted:
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "age": age,
                "area": area,
                "gender": gender,
                "education": education,
                "years_experience": years_experience,
                "marital_status": marital_status,
                "socio_professional_group": socio_group
            }
        )
        if response.status_code == 200:
            st.success(f"Revenu prédit : {response.json()['predicted_income']} DH")
        else:
            st.error("Erreur de prédiction")