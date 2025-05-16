from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# REMOVE any CustomPreprocessor class definition from here

# Chargement du modèle et des composants de prétraitement
try:
    model_data = joblib.load('best_mlp_model.pkl')
    model = model_data['model']
    components = model_data['preprocessor_components'] # This is now a dictionary

    num_imputer = components['num_imputer']
    cat_imputer = components['cat_imputer']
    scaler = components['scaler']
    # These are the column names the model was trained on (after all preprocessing)
    train_cols_final = components['train_cols_after_dummies']
    # These are the original column names before any transformation
    original_numerical_cols = components['original_numerical_cols']
    original_categorical_cols = components['original_categorical_cols']

except Exception as e:
    print(f"Error loading model or preprocessor components: {e}")
    model = None
    num_imputer, cat_imputer, scaler, train_cols_final, original_numerical_cols, original_categorical_cols = [None] * 6


class InputFeatures(BaseModel):
    Age: int
    Sexe: str
    Milieu: str
    Niveau_education: str
    Annees_experience: int
    Etat_matrimonial: str
    CSP: str
    Secteur_emploi: str # Can be NaN if 'Inactifs'
    Propriete_immobiliere: str
    Vehicule_motorise: str
    Terrain_agricole: str
    Revenu_secondaire: str
    Region_geographique: str

@app.post("/predict")
async def predict_income(features: InputFeatures):
    if not all([model, num_imputer, cat_imputer, scaler, train_cols_final, original_numerical_cols, original_categorical_cols]):
        return {"error": "Model or preprocessor components not loaded correctly."}

    input_df = pd.DataFrame([features.dict()])
    X_new = input_df.copy()

    # --- Manual Preprocessing Steps ---
    # Identify current numerical and categorical columns from input based on original lists
    current_num_cols = [col for col in original_numerical_cols if col in X_new.columns]
    current_cat_cols = [col for col in original_categorical_cols if col in X_new.columns]

    # 1. Imputation
    if current_num_cols:
        X_new[current_num_cols] = num_imputer.transform(X_new[current_num_cols])
    if current_cat_cols:
        X_new[current_cat_cols] = cat_imputer.transform(X_new[current_cat_cols])

    # 2. One-Hot Encoding for categorical columns
    if current_cat_cols:
        X_new = pd.get_dummies(X_new, columns=current_cat_cols, prefix=current_cat_cols, dummy_na=False)

    # 3. Align columns with those seen during training (train_cols_final)
    # Add missing columns (that were in training) and fill with 0
    for col in train_cols_final:
        if col not in X_new.columns:
            X_new[col] = 0
    # Ensure order and remove any extra columns not in training_cols_final
    X_new = X_new.reindex(columns=train_cols_final, fill_value=0)
    
    # 4. Scaling for original numerical columns (their names are preserved through one-hot encoding)
    # We need to scale only the original numerical columns that are still present by their original name
    cols_to_scale_now = [col for col in original_numerical_cols if col in X_new.columns]
    if cols_to_scale_now:
        X_new[cols_to_scale_now] = scaler.transform(X_new[cols_to_scale_now])
    # --- End of Manual Preprocessing ---
    
    processed_data = X_new
    prediction = model.predict(processed_data)
    
    return {"predicted_income": round(float(prediction[0]), 2)}