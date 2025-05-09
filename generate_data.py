import pandas as pd
import numpy as np
import random
import uuid # Pour CIN et Email

# --- Paramètres Généraux ---
N_RECORDS = 40000
FILENAME = "dataset_revenu_marocains.csv"

# --- Constantes Statistiques (Cibles HCP) ---
REVENU_ANNUEL_MOYEN_GLOBAL_CIBLE = 21949
REVENU_ANNUEL_MOYEN_URBAIN_CIBLE = 26988
REVENU_ANNUEL_MOYEN_RURAL_CIBLE = 12862

# Calcul de la proportion urbain/rural pour atteindre les moyennes
# p_urbain * REVENU_URBAIN + (1-p_urbain) * REVENU_RURAL = REVENU_GLOBAL
# p_urbain = (REVENU_GLOBAL - REVENU_RURAL) / (REVENU_URBAIN - REVENU_RURAL)
P_URBAIN = (REVENU_ANNUEL_MOYEN_GLOBAL_CIBLE - REVENU_ANNUEL_MOYEN_RURAL_CIBLE) / \
           (REVENU_ANNUEL_MOYEN_URBAIN_CIBLE - REVENU_ANNUEL_MOYEN_RURAL_CIBLE)
P_RURAL = 1 - P_URBAIN

# --- Définition des Catégories ---
MILIEU_OPTS = ['Urbain', 'Rural']
SEXE_OPTS = ['Homme', 'Femme']
NIVEAU_EDUCATION_OPTS = ['Sans niveau', 'Fondamental', 'Secondaire', 'Supérieur']
ETAT_MATRIMONIAL_OPTS = ['Célibataire', 'Marié', 'Divorcé', 'Veuf']
CSP_OPTS = ['Cadres supérieurs', 'Professions intermédiaires', 'Employés', 'Ouvriers', 'Agriculteurs', 'Inactifs']
REGION_GEO_OPTS = ['Nord', 'Centre', 'Sud', 'Est', 'Ouest'] # Centre, Ouest, Nord plus riches
SECTEUR_EMPLOI_OPTS = ['Public', 'Privé', 'Informel']
OUI_NON_OPTS = ['Oui', 'Non']

# --- Fonctions de Génération des Caractéristiques ---

def generate_age(n):
    # Revenu tend à augmenter avec l'âge, approchant la retraite [18-63]
    return np.random.randint(18, 64, n)

def generate_milieu(n):
    return np.random.choice(MILIEU_OPTS, n, p=[P_URBAIN, P_RURAL])

def generate_sexe(n):
    # Revenu moyen des hommes généralement plus élevé
    return np.random.choice(SEXE_OPTS, n, p=[0.52, 0.48]) # Légère prédominance pour illustrer écart

def generate_niveau_education(n):
    # Revenu plus élevé avec niveau d'éducation supérieur
    # Pondération : Moins de 'Sans niveau' et 'Supérieur', plus de 'Fondamental' et 'Secondaire'
    return np.random.choice(NIVEAU_EDUCATION_OPTS, n, p=[0.15, 0.35, 0.35, 0.15])

def generate_annees_experience(age, niveau_education):
    experience = []
    for a, edu in zip(age, niveau_education):
        min_age_travail = 18
        if edu == 'Supérieur':
            min_age_travail = 23
        elif edu == 'Secondaire':
            min_age_travail = 20
        
        max_exp = a - min_age_travail
        if max_exp < 0: max_exp = 0
        
        # Expérience plausible
        exp = np.random.randint(0, max_exp + 1) if max_exp > 0 else 0
        # Assurer que l'expérience ne dépasse pas age - 16 (pour un début à 16 ans min)
        exp = min(exp, a - 16) if a > 16 else 0
        experience.append(max(0,exp)) # Assurer non-négatif
    return np.array(experience)

def generate_etat_matrimonial(n, age):
    # Impact variable. Hommes mariés, divorcés, veufs gagnent plus. Femmes divorcées/veuves aussi.
    # Simplification: plus de mariés avec l'âge
    etats = []
    for a in age:
        if a < 25:
            etats.append(np.random.choice(ETAT_MATRIMONIAL_OPTS, p=[0.8, 0.15, 0.03, 0.02])) # Majorité célibataire
        elif a < 45:
            etats.append(np.random.choice(ETAT_MATRIMONIAL_OPTS, p=[0.2, 0.65, 0.1, 0.05])) # Majorité marié
        else:
            etats.append(np.random.choice(ETAT_MATRIMONIAL_OPTS, p=[0.1, 0.55, 0.15, 0.2])) # Plus de divorcés/veufs
    return np.array(etats)

def generate_csp(niveau_education, annees_experience, age):
    # Classés du plus haut revenu au plus bas.
    # 'Inactifs' pour certains cas (ex: très jeunes sans exp, ou plus âgés)
    csps = []
    for edu, exp, a in zip(niveau_education, annees_experience, age):
        if exp < 1 and a < 22 and edu in ['Sans niveau', 'Fondamental']:
            csps.append('Inactifs')
        elif edu == 'Supérieur':
            if exp > 10: csps.append('Cadres supérieurs')
            elif exp > 3: csps.append('Professions intermédiaires')
            else: csps.append('Employés')
        elif edu == 'Secondaire':
            if exp > 15: csps.append('Professions intermédiaires')
            elif exp > 5: csps.append('Employés')
            else: csps.append('Ouvriers')
        elif edu == 'Fondamental':
            if exp > 10: csps.append('Ouvriers')
            else: csps.append(np.random.choice(['Ouvriers', 'Agriculteurs'], p=[0.7,0.3]))
        else: # Sans niveau
            csps.append(np.random.choice(['Ouvriers', 'Agriculteurs', 'Inactifs'], p=[0.4,0.4,0.2]))
    return np.array(csps)

def generate_possession_biens(n, csp_list, milieu_list):
    # Corrélation positive avec revenu/CSP
    prop_immo, veh_motor, terr_agri = [], [], []
    for csp, milieu in zip(csp_list, milieu_list):
        p_immo, p_veh, p_agri = 0.1, 0.1, 0.05 # Base pour Inactifs/Bas CSP

        if csp == 'Cadres supérieurs': p_immo, p_veh = 0.8, 0.9
        elif csp == 'Professions intermédiaires': p_immo, p_veh = 0.6, 0.7
        elif csp == 'Employés': p_immo, p_veh = 0.3, 0.4
        elif csp == 'Ouvriers': p_immo, p_veh = 0.15, 0.25
        elif csp == 'Agriculteurs': p_immo, p_veh, p_agri = 0.4, 0.3, 0.7
        
        if milieu == 'Rural' and csp == 'Agriculteurs': p_agri = 0.85
        if milieu == 'Rural': p_veh *= 0.8 # Moins de véhicules en rural sauf si agriculteur

        prop_immo.append(np.random.choice(OUI_NON_OPTS, p=[p_immo, 1-p_immo]))
        veh_motor.append(np.random.choice(OUI_NON_OPTS, p=[p_veh, 1-p_veh]))
        terr_agri.append(np.random.choice(OUI_NON_OPTS, p=[p_agri, 1-p_agri]))
    return prop_immo, veh_motor, terr_agri

def generate_region_geographique(n):
    # Centre, Ouest, Nord plus riches
    return np.random.choice(REGION_GEO_OPTS, n, p=[0.25, 0.30, 0.15, 0.15, 0.15]) # Nord, Centre, Sud, Est, Ouest

def generate_secteur_emploi(n, csp_list):
    secteurs = []
    for csp in csp_list:
        if csp == 'Inactifs':
            secteurs.append(np.nan) # Pas de secteur si inactif
        elif csp == 'Agriculteurs':
            secteurs.append(np.random.choice(['Privé', 'Informel'], p=[0.3,0.7]))
        elif csp in ['Cadres supérieurs', 'Professions intermédiaires']:
            secteurs.append(np.random.choice(['Public', 'Privé'], p=[0.4, 0.6]))
        elif csp == 'Employés':
            secteurs.append(np.random.choice(['Public', 'Privé', 'Informel'], p=[0.3, 0.5, 0.2]))
        else: # Ouvriers
            secteurs.append(np.random.choice(['Privé', 'Informel'], p=[0.4, 0.6]))
    return np.array(secteurs)

def generate_revenu_secondaire(n, csp_list):
    # Plus probable pour CSP élevés ou certains secteurs
    probs = []
    for csp in csp_list:
        if csp in ['Cadres supérieurs', 'Professions intermédiaires']:
            probs.append(0.4)
        elif csp == 'Agriculteurs':
            probs.append(0.3)
        elif csp == 'Employés':
            probs.append(0.2)
        else:
            probs.append(0.1)
    return np.array([np.random.choice(OUI_NON_OPTS, p=[p, 1-p]) for p in probs])


def generate_revenu_annuel(df):
    # Distributions séparées pour mieux contrôler les moyennes et proportions
    urbain_mask = df['Milieu'] == 'Urbain'
    rural_mask = ~urbain_mask

    revenus = np.zeros(len(df))

    # Ajustement des paramètres sigma pour viser les contraintes de répartition
    urbain_mu, urbain_sigma = np.log(7300), 0.5  # Ajusté pour cible 65.9%
    rural_mu, rural_sigma = np.log(2000), 2.11  # Ajusté pour cible 85.4%

    revenus_urbains = np.random.lognormal(mean=urbain_mu, sigma=urbain_sigma, size=sum(urbain_mask))
    revenus_ruraux = np.random.lognormal(mean=rural_mu, sigma=rural_sigma, size=sum(rural_mask))

    revenus[urbain_mask] = revenus_urbains
    revenus[rural_mask] = revenus_ruraux

    # Application des facteurs
    for i in range(len(df)):
        rev = revenus[i]
        record = df.iloc[i]

        # Sexe
        if record['Sexe'] == 'Homme': rev *= 1.15

        # Niveau d'éducation
        if record['Niveau_education'] == 'Supérieur': 
            rev *= 1.8
            if record['Milieu'] == 'Rural': rev *= 0.8  # Renforcement de la réduction
        elif record['Niveau_education'] == 'Secondaire': 
            rev *= 1.3
            if record['Milieu'] == 'Rural': rev *= 0.85  # Renforcement de la réduction
        elif record['Niveau_education'] == 'Fondamental': 
            rev *= 1.05
        else: 
            rev *= 0.85

        # Années d'expérience - impact réduit en milieu rural
        annees_exp_val = record['Annees_experience'] if pd.notna(record['Annees_experience']) else 0
        exp_factor = 0.01 if record['Milieu'] == 'Urbain' else 0.0025  # Réduit davantage
        rev *= (1 + annees_exp_val * exp_factor)

        # CSP avec différenciation urbain/rural plus marquée
        if record['CSP'] == 'Cadres supérieurs': 
            rev *= 2.0
            if record['Milieu'] == 'Rural': rev *= 0.75  # Renforcement de la réduction
        elif record['CSP'] == 'Professions intermédiaires': 
            rev *= 1.6
            if record['Milieu'] == 'Rural': rev *= 0.8
        elif record['CSP'] == 'Employés': 
            rev *= 1.15
            if record['Milieu'] == 'Rural': rev *= 0.9
        elif record['CSP'] == 'Ouvriers': 
            rev *= 0.9
            if record['Milieu'] == 'Rural': rev *= 0.85
        elif record['CSP'] == 'Agriculteurs': 
            rev *= 0.85
            if record['Milieu'] == 'Rural': rev *= 0.75
        elif record['CSP'] == 'Inactifs': 
            rev = np.random.uniform(300, 2000)
            if record['Milieu'] == 'Rural': rev *= 0.6

        # Région géographique
        if record['Region_geographique'] in ['Centre', 'Ouest']: 
            rev *= 1.15
            if record['Milieu'] == 'Rural': rev *= 0.95
        elif record['Region_geographique'] == 'Nord': 
            rev *= 1.1
            if record['Milieu'] == 'Rural': rev *= 0.95
        elif record['Region_geographique'] in ['Sud', 'Est']: rev *= 0.9

        # Secteur d'emploi
        if pd.notna(record['Secteur_emploi']):
            if record['Secteur_emploi'] == 'Public': rev *= 1.05
            elif record['Secteur_emploi'] == 'Privé': rev *= 1.15
            elif record['Secteur_emploi'] == 'Informel': 
                rev *= 0.8
                if record['Milieu'] == 'Rural': rev *= 0.85

        # Revenu secondaire avec impact réduit en milieu rural
        if record['Revenu_secondaire'] == 'Oui': 
            mult = 1.2 if record['Milieu'] == 'Urbain' else 1.05
            rev *= mult

        # Possession de biens
        if record['Propriete_immobiliere'] == 'Oui': rev *= 1.03
        if record['Vehicule_motorise'] == 'Oui': rev *= 1.02
        if record['Terrain_agricole'] == 'Oui' and record['CSP'] == 'Agriculteurs': 
            mult = 1.05 if record['Milieu'] == 'Urbain' else 1.01
            rev *= mult

        # Plafonnement et plancher
        rev = max(300, min(rev, 600000))
        if record['CSP'] == 'Inactifs': 
            max_revenu = 8000 if record['Milieu'] == 'Urbain' else 4000
            rev = max(0, min(rev, max_revenu))

        revenus[i] = round(rev, 0)

    # Nous ne faisons plus l'ajustement final ici, il sera fait après l'ajout des valeurs aberrantes
    return revenus

# --- Fonctions pour Imperfections ---

def add_valeurs_manquantes(df, colonnes_pour_nan, p_nan=0.015):  # AJUSTÉ: p_nan réduit à 1.5%
    for col in colonnes_pour_nan:
        if col in df.columns:
            mask = np.random.choice([True, False], size=len(df), p=[p_nan, 1-p_nan])
            df.loc[mask, col] = np.nan
    return df

def add_valeurs_aberrantes(df):
    # Age (quelques très jeunes/âgés)
    n_aberr_age = int(0.005 * len(df))  # Maintenu à 0.5%
    for _ in range(n_aberr_age):
        idx = np.random.randint(0, len(df))
        df.loc[idx, 'Age'] = np.random.choice([15, 16, 70, 75, 80])
        # Ajuster expérience si âge devient trop bas pour l'expérience existante
        current_exp = df.loc[idx, 'Annees_experience']
        if pd.notna(current_exp) and df.loc[idx, 'Age'] < current_exp + 16:
            df.loc[idx, 'Annees_experience'] = max(0, df.loc[idx, 'Age'] - 16 - np.random.randint(0, 3))

    # Revenu Annuel (quelques très hauts/bas)
    n_aberr_revenu = int(0.004 * len(df))  # AJUSTÉ: réduit à 0.4%
    for _ in range(n_aberr_revenu):
        idx = np.random.randint(0, len(df))
        if df.loc[idx, 'CSP'] != 'Inactifs':  # Ne pas mettre revenu aberrant pour inactifs
            df.loc[idx, 'Revenu_Annuel'] = np.random.choice([100, 200, 300000, 400000])  # AJUSTÉ: valeurs réduites

    # Années d'expérience (quelques valeurs illogiques par rapport à l'âge, mais pas négatives)
    n_aberr_exp = int(0.002 * len(df))  # AJUSTÉ: réduit à 0.2%
    for _ in range(n_aberr_exp):
        idx = np.random.randint(0, len(df))
        age_val = df.loc[idx, 'Age']
        df.loc[idx, 'Annees_experience'] = age_val - 10
        if df.loc[idx, 'Annees_experience'] < 0:
            df.loc[idx, 'Annees_experience'] = 0

    return df

# --- Génération du Dataset ---
def generate_dataset():
    data = pd.DataFrame()

    # Caractéristiques Principales
    data['Age'] = generate_age(N_RECORDS)
    data['Milieu'] = generate_milieu(N_RECORDS)
    data['Sexe'] = generate_sexe(N_RECORDS)
    data['Niveau_education'] = generate_niveau_education(N_RECORDS)
    data['Annees_experience'] = generate_annees_experience(data['Age'], data['Niveau_education'])
    data['Etat_matrimonial'] = generate_etat_matrimonial(N_RECORDS, data['Age'])
    data['CSP'] = generate_csp(data['Niveau_education'], data['Annees_experience'], data['Age'])
    
    prop_immo, veh_motor, terr_agri = generate_possession_biens(N_RECORDS, data['CSP'], data['Milieu'])
    data['Propriete_immobiliere'] = prop_immo
    data['Vehicule_motorise'] = veh_motor
    data['Terrain_agricole'] = terr_agri

    # Caractéristiques Additionnelles
    data['Region_geographique'] = generate_region_geographique(N_RECORDS)
    data['Secteur_emploi'] = generate_secteur_emploi(N_RECORDS, data['CSP'])
    data['Revenu_secondaire'] = generate_revenu_secondaire(N_RECORDS, data['CSP'])
    
    # Revenu Annuel (dépendant des autres)
    data['Revenu_Annuel'] = generate_revenu_annuel(data.copy())

    # Imperfections
    # Colonnes Redondantes
    data['Revenu_Mensuel'] = (data['Revenu_Annuel'] / 12).round(2)
    
    bins = [18, 25, 45, 60, 100] # 100 pour couvrir les âges aberrants
    labels = ['Jeune', 'Adulte', 'Senior', 'Âgé']
    data['Categorie_age'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

    # Colonnes Non Pertinentes
    data['Adresse_Email'] = [f"user{uuid.uuid4().hex[:6]}@example.com" for _ in range(N_RECORDS)]
    data['CIN'] = [f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000,999999)}" for _ in range(N_RECORDS)]

    # Valeurs Manquantes (après génération de toutes les colonnes de base)
    cols_with_nan = ['Etat_matrimonial', 'Secteur_emploi', 'Revenu_secondaire', 
                     'Propriete_immobiliere', 'Vehicule_motorise', 'Terrain_agricole',
                     'Annees_experience']
    data = add_valeurs_manquantes(data, cols_with_nan, p_nan=0.015)

    # Valeurs Aberrantes (après génération et NaN, pour ne pas écraser trop de NaN)
    data = add_valeurs_aberrantes(data)
    
    # Ajustement final des moyennes APRÈS l'ajout des valeurs aberrantes
    urbain_mask = data['Milieu'] == 'Urbain'
    rural_mask = ~urbain_mask
    
    urbain_revenus_actuels = data.loc[urbain_mask, 'Revenu_Annuel'].dropna()
    rural_revenus_actuels = data.loc[rural_mask, 'Revenu_Annuel'].dropna()
    
    if not urbain_revenus_actuels.empty:
        mean_urbain_actuel = urbain_revenus_actuels.mean()
        if mean_urbain_actuel != 0 and pd.notna(mean_urbain_actuel):
            factor_urbain = REVENU_ANNUEL_MOYEN_URBAIN_CIBLE / mean_urbain_actuel
            data.loc[urbain_mask & data['Revenu_Annuel'].notna(), 'Revenu_Annuel'] *= factor_urbain
    
    if not rural_revenus_actuels.empty:
        mean_rural_actuel = rural_revenus_actuels.mean()
        if mean_rural_actuel != 0 and pd.notna(mean_rural_actuel):
            factor_rural = REVENU_ANNUEL_MOYEN_RURAL_CIBLE / mean_rural_actuel
            data.loc[rural_mask & data['Revenu_Annuel'].notna(), 'Revenu_Annuel'] *= factor_rural
    
    # --- Ajustement additionnel pour respecter la contrainte sur le pourcentage urbain ---
    target_pct = 65.7
    tol = 0.1  # tolérance de 0.1%
    max_iterations = 10

    for _ in range(max_iterations):
        urban_values = data.loc[urbain_mask, 'Revenu_Annuel']
        urban_mean = urban_values.mean()
        pct_urban = (urban_values < urban_mean).mean() * 100

        if abs(pct_urban - target_pct) < tol:
            break

        if pct_urban > target_pct:
            # Trop d'individus en dessous de la moyenne : on augmente les bas revenus
            factor = 1 + (pct_urban - target_pct) / 100  
            data.loc[(urbain_mask) & (data['Revenu_Annuel'] < urban_mean), 'Revenu_Annuel'] *= factor
        else:
            # Pas assez d'individus en dessous de la moyenne : on réduit les revenus supérieurs ou égaux à la moyenne
            factor = 1 - (target_pct - pct_urban) / 100  
            data.loc[(urbain_mask) & (data['Revenu_Annuel'] >= urban_mean), 'Revenu_Annuel'] *= factor

        # Ré-ajuster pour maintenir la moyenne cible urbaine
        urban_mean_new = data.loc[urbain_mask, 'Revenu_Annuel'].mean()
        if urban_mean_new != 0:
            factor_urban_correction = REVENU_ANNUEL_MOYEN_URBAIN_CIBLE / urban_mean_new
            data.loc[urbain_mask, 'Revenu_Annuel'] *= factor_urban_correction
    # --- Fin de l'ajustement additionnel ---

    data['Revenu_Annuel'] = data['Revenu_Annuel'].round(0)
    data['Revenu_Mensuel'] = (data['Revenu_Annuel'] / 12).round(2)
    
    # Vérification finale de cohérence pour Annees_experience vs Age
    mask_exp_incoherent = data['Annees_experience'] > (data['Age'] - 16)
    data.loc[mask_exp_incoherent, 'Annees_experience'] = (data.loc[mask_exp_incoherent, 'Age'] - 16 - np.random.randint(0,2, size=mask_exp_incoherent.sum())).clip(lower=0)
    
    # S'assurer que les inactifs ont bien un revenu bas ou NaN si le revenu est manquant
    inactifs_mask = data['CSP'] == 'Inactifs'
    data.loc[inactifs_mask, 'Revenu_Annuel'] = data.loc[inactifs_mask, 'Revenu_Annuel'].apply(lambda x: min(x, 10000) if pd.notna(x) else x)
    data.loc[inactifs_mask, 'Revenu_Mensuel'] = (data.loc[inactifs_mask, 'Revenu_Annuel'] / 12).round(2)

    # Ordonner les colonnes pour la lisibilité (optionnel)
    ordered_columns = [
        'Age', 'Categorie_age', 'Sexe', 'Milieu', 'Region_geographique', 'Etat_matrimonial',
        'Niveau_education', 'Annees_experience', 'CSP', 'Secteur_emploi',
        'Propriete_immobiliere', 'Vehicule_motorise', 'Terrain_agricole',
        'Revenu_secondaire', 'Revenu_Annuel', 'Revenu_Mensuel',
        'Adresse_Email', 'CIN'
    ]
    # S'assurer que toutes les colonnes sont présentes avant de réordonner
    final_columns = [col for col in ordered_columns if col in data.columns]
    missing_cols = [col for col in data.columns if col not in final_columns]
    data = data[final_columns + missing_cols]

    return data

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Génération de {N_RECORDS} enregistrements...")
    dataset = generate_dataset()
    
      
    #Vérification finale des contraintes statistiques clés
    print("\n*** VÉRIFICATION FINALE DES CONTRAINTES ***")
    mean_global_final = dataset['Revenu_Annuel'].mean()
    mean_urbain_final = dataset[dataset['Milieu'] == 'Urbain']['Revenu_Annuel'].mean()
    mean_rural_final = dataset[dataset['Milieu'] == 'Rural']['Revenu_Annuel'].mean()
    
    print(f"Revenu Global: {mean_global_final:.1f} DH (Cible: {REVENU_ANNUEL_MOYEN_GLOBAL_CIBLE} DH)")
    print(f"Revenu Urbain: {mean_urbain_final:.1f} DH (Cible: {REVENU_ANNUEL_MOYEN_URBAIN_CIBLE} DH)")
    print(f"Revenu Rural: {mean_rural_final:.1f} DH (Cible: {REVENU_ANNUEL_MOYEN_RURAL_CIBLE} DH)")
    
    pct_inf_moy_global_final = (dataset['Revenu_Annuel'] < mean_global_final).mean() * 100
    pct_inf_moy_urbain_final = (dataset[dataset['Milieu'] == 'Urbain']['Revenu_Annuel'] < mean_urbain_final).mean() * 100
    pct_inf_moy_rural_final = (dataset[dataset['Milieu'] == 'Rural']['Revenu_Annuel'] < mean_rural_final).mean() * 100
    
    print(f"% sous moyenne globale: {pct_inf_moy_global_final:.1f}% (Cible: 71.8%)")
    print(f"% sous moyenne urbaine: {pct_inf_moy_urbain_final:.1f}% (Cible: 65.9%)")
    print(f"% sous moyenne rurale: {pct_inf_moy_rural_final:.1f}% (Cible: 85.4%)")
    
    dataset.to_csv(FILENAME, index=False, encoding='utf-8')
    print(f"\nDataset '{FILENAME}' généré avec succès ({len(dataset)} enregistrements).")
