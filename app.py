import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
from time import sleep

# Set page configuration
st.set_page_config(
    page_title="Prédiction de Revenu Marocain",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to preprocess input data using loaded components
def preprocess_input(input_df, preprocessor_components):
    """
    Preprocesses the input DataFrame using the loaded preprocessor components.
    """
    # Ensure input_df is a DataFrame
    if not isinstance(input_df, pd.DataFrame):
        input_df = pd.DataFrame(input_df, index=[0])

    df = input_df.copy()

    original_numerical_cols = preprocessor_components.get('original_numerical_cols', [])
    original_categorical_cols = preprocessor_components.get('original_categorical_cols', [])
    num_imputer = preprocessor_components.get('num_imputer')
    cat_imputer = preprocessor_components.get('cat_imputer')
    scaler = preprocessor_components.get('scaler')
    train_cols_after_dummies = preprocessor_components.get('train_cols_after_dummies', [])

    if not all([num_imputer, cat_imputer, scaler, train_cols_after_dummies]):
        raise ValueError("Un ou plusieurs composants de prétraitement sont manquants ou invalides.")

    # Impute numerical
    if original_numerical_cols:
        num_cols_in_df = [col for col in original_numerical_cols if col in df.columns]
        if num_cols_in_df:
            df[num_cols_in_df] = num_imputer.transform(df[num_cols_in_df])

    # Impute categorical
    if original_categorical_cols:
        cat_cols_in_df = [col for col in original_categorical_cols if col in df.columns]
        if cat_cols_in_df:
            df[cat_cols_in_df] = cat_imputer.transform(df[cat_cols_in_df])

    # One-hot encode categorical
    if original_categorical_cols:
        cat_cols_for_dummies = [col for col in original_categorical_cols if col in df.columns]
        if cat_cols_for_dummies:
            df = pd.get_dummies(df, columns=cat_cols_for_dummies, prefix=cat_cols_for_dummies, dummy_na=False)

    # Align columns with training data after one-hot encoding
    df = df.reindex(columns=train_cols_after_dummies, fill_value=0)

    # Scale numerical features
    # original_numerical_cols refers to names before any encoding.
    # These columns should still exist if they were not part of categorical_cols.
    if original_numerical_cols:
        # We scale the columns that were identified as numerical at the start and are present in the final feature set
        cols_to_scale_in_df = [col for col in original_numerical_cols if col in df.columns]
        if cols_to_scale_in_df:
            df[cols_to_scale_in_df] = scaler.transform(df[cols_to_scale_in_df])
            
    # Ensure final columns are exactly as expected by the model, in the correct order
    df = df[train_cols_after_dummies]

    return df

# Load the model at app startup
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model_data = joblib.load('best_mlp_model.pkl')
        model = model_data['model']
        
        if 'preprocessor_components' not in model_data:
            return model_data, model, None, "Clé 'preprocessor_components' non trouvée dans le fichier modèle. Veuillez vérifier la sauvegarde du modèle."

        preprocessor_components = model_data['preprocessor_components']
        
        required_keys = ['num_imputer', 'cat_imputer', 'scaler', 'train_cols_after_dummies', 'original_numerical_cols', 'original_categorical_cols']
        if not all(key in preprocessor_components for key in required_keys):
            missing = [key for key in required_keys if key not in preprocessor_components]
            return model_data, model, None, f"Modèle chargé, mais 'preprocessor_components' est incomplet. Clés manquantes: {missing}. Vérifiez la sauvegarde du modèle."
            
        return model_data, model, preprocessor_components, None
    except FileNotFoundError:
        return {'metadata': {'performance': {'R2': 0, 'MAE': 0, 'RMSE': 0}}}, None, None, "Modèle 'best_mlp_model.pkl' non trouvé. Assurez-vous qu'il existe dans le répertoire."
    except KeyError as e:
        return {'metadata': {'performance': {'R2': 0, 'MAE': 0, 'RMSE': 0}}}, None, None, f"Erreur lors du chargement du modèle : clé manquante '{str(e)}' dans 'best_mlp_model.pkl'. Vérifiez la sauvegarde du modèle."
    except Exception as e:
        return {'metadata': {'performance': {'R2': 0, 'MAE': 0, 'RMSE': 0}}}, None, None, f"Erreur lors du chargement du modèle : {str(e)}"

model_data, model, preprocessor_components, load_error = load_model_and_preprocessor()

if load_error:
    st.error(load_error)

st.title("💰 Prédiction de Revenu Annuel Marocain")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("Informations Personnelles")
    age = st.slider("Âge", 18, 65, 30)
    sexe = st.selectbox("Sexe", ["Homme", "Femme"])
    etat_matrimonial = st.selectbox("État matrimonial", ["Célibataire", "Marié", "Divorcé", "Veuf"])
    milieu = st.selectbox("Milieu de vie", ["Urbain", "Rural"])
    region = st.selectbox("Région géographique", ["Nord", "Centre", "Sud", "Est", "Ouest"])

with col2:
    st.header("Caractéristiques Professionnelles")
    experience = st.number_input("Années d'expérience", 0, 50, 5)
    education = st.selectbox("Niveau d'éducation", 
                          ["Sans niveau", "Fondamental", "Secondaire", "Supérieur"])
    csp = st.selectbox("Catégorie Socio-Professionnelle",
                      ["Agriculteurs", "Ouvriers", "Employés", 
                       "Professions intermédiaires", "Cadres supérieurs", "Inactifs"])
    secteur = st.selectbox("Secteur d'emploi", ["Public", "Privé", "Informel"]) # Assuming 'Inactifs' will have NaN or specific handling for S.E.

# Patrimoine
st.header("Patrimoine")
col3, col4, col5 = st.columns(3)

with col3:
    proprio = st.selectbox("Propriété immobilière", ["Oui", "Non"])
    
with col4:
    vehicule = st.selectbox("Véhicule motorisé", ["Oui", "Non"])
    
with col5:
    terrain = st.selectbox("Terrain agricole", ["Oui", "Non"])

revenu_sec = st.selectbox("Revenu secondaire", ["Oui", "Non"])

# Bouton de prédiction
if st.button("Estimer le Revenu"):
    payload = {
        "Age": age,
        "Sexe": sexe,
        "Milieu": milieu,
        "Niveau_education": education,
        "Annees_experience": experience,
        "Etat_matrimonial": etat_matrimonial,
        "CSP": csp,
        # Handle 'Secteur_emploi' for 'Inactifs' CSP if necessary.
        # The preprocessing in the notebook imputes NaNs, so sending the selected value is fine.
        # If CSP is 'Inactifs', 'Secteur_emploi' might be less relevant or NaN in training.
        "Secteur_emploi": secteur if csp != "Inactifs" else np.nan, # Or send as is and let imputer handle
        "Propriete_immobiliere": proprio,
        "Vehicule_motorise": vehicule,
        "Terrain_agricole": terrain,
        "Revenu_secondaire": revenu_sec,
        "Region_geographique": region
    }
    
    with st.spinner("Calcul en cours..."):
        success = False
        api_working = True # Assume API might be used, though focus is local
        prediction = None
        
        # Option 1: Prédiction via API FastAPI (Kept for completeness, but local is primary focus now)
        try:
            response = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
            if response.status_code == 200:
                prediction = response.json()["predicted_income"]
                success = True
            else:
                st.error(f"Erreur lors de la prédiction via API: {response.status_code}")
                if response.status_code == 500:
                    st.info("Le serveur API a rencontré une erreur interne. Vérifiez les logs du serveur.")
                api_working = False
        except requests.exceptions.ConnectionError:
            st.warning("L'API n'est pas accessible. Tentative de prédiction locale...")
            api_working = False
        except Exception as api_error:
            st.error(f"Erreur API: {str(api_error)}")
            api_working = False
            
        # Option 2: Prédiction locale
        if not api_working: # If API failed or was skipped
            if model is not None and preprocessor_components is not None:
                try:
                    input_df = pd.DataFrame([payload])
                    processed_data = preprocess_input(input_df, preprocessor_components)
                    prediction_array = model.predict(processed_data)
                    prediction = prediction_array[0] if isinstance(prediction_array, np.ndarray) else prediction_array
                    success = True
                except ValueError as ve: # Catch specific errors from preprocess_input
                    st.error(f"Erreur de prétraitement des données: {str(ve)}")
                except Exception as local_error:
                    st.error(f"Erreur de prédiction locale: {str(local_error)}")
                    st.info("Vérifiez que les données d'entrée sont correctes et que le modèle et ses composants sont valides.")
            else:
                st.error("Impossible de faire une prédiction locale - modèle ou composants de préprocesseur non chargés.")
                    
    # Display prediction if successful
    if success and prediction is not None:
        st.markdown(f"""
        <div class="success-box">
            <h3>Revenu annuel estimé : {prediction:,.2f} DH</h3>
            <p><em>Soit environ {prediction/12:,.2f} DH/mois</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display insights based on prediction
        if prediction > 100000:
            st.info("Ce profil correspond à un revenu élevé, probablement associé à un niveau d'éducation supérieur et une position cadre.")
        elif prediction > 50000:
            st.info("Ce profil correspond à un revenu moyen, typique des professions intermédiaires.")
        else:
            st.info("Ce profil correspond à un revenu modeste, potentiellement associé au secteur informel ou aux activités agricoles.")
    elif not success and not load_error : # If prediction failed for other reasons than model load
        st.error("La prédiction n'a pas pu être effectuée.")


# Section supplémentaire
st.markdown("---")
st.subheader("📊 Analyse des Performances du Modèle Sauvegardé")
if isinstance(model_data, dict) and 'metadata' in model_data and 'performance' in model_data['metadata']:
    metrics = model_data['metadata']['performance']
    
    met1, met2, met3 = st.columns(3)
    with met1:
        r2 = metrics.get('R2', 0)
        st.metric("Précision du modèle (R²)", f"{r2:.2%}" if isinstance(r2, (int, float)) else "N/A")
        
    with met2:
        mae = metrics.get('MAE', 0)
        st.metric("Marge d'erreur moyenne", f"{mae:,.2f} DH" if isinstance(mae, (int, float)) else "N/A")
        
    with met3:
        rmse = metrics.get('RMSE', 0)
        st.metric("Erreur quadratique", f"{rmse:,.2f} DH" if isinstance(rmse, (int, float)) else "N/A")
else:
    st.warning("Informations sur les performances du modèle non disponibles ou format incorrect.")

# Ajout d'une note explicative
st.markdown("---")
st.markdown("""
### À propos de ce modèle
Ce modèle de prédiction utilise des techniques d'apprentissage automatique (Réseau de Neurones MLP) pour estimer le revenu annuel 
d'une personne au Maroc en fonction de ses caractéristiques socio-démographiques et professionnelles.

Le modèle a été entraîné sur un dataset synthétique représentant les tendances socio-économiques marocaines.
Les hyperparamètres du modèle MLP ont été optimisés pour obtenir les meilleures performances.

*Note : Ce modèle est fourni à titre indicatif uniquement et ne constitue pas un conseil financier.*
""")

# Sidebar for additional information
with st.sidebar:
    st.title("Informations Complémentaires")
    st.markdown("""
    ### Utilisation
    1. Renseignez vos informations personnelles et professionnelles.
    2. Cliquez sur "Estimer le Revenu".
    3. Consultez le résultat de la prédiction.
    
    ### Variables importantes (généralement)
    - Niveau d'éducation
    - Années d'expérience
    - Catégorie socio-professionnelle (CSP)
    - Milieu de vie (Urbain/Rural)
    
    *(L'importance réelle des variables peut être analysée plus en détail à partir du modèle entraîné).*
    
    ### Contact
    Pour toute question ou suggestion, veuillez nous contacter :
    - Email: contact@example.com 
    """)