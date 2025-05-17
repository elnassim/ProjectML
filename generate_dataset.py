import pandas as pd
import numpy as np
import random
import uuid  # Ajout de l'import uuid pour générer des adresses email uniques

# --- Paramètres Généraux ---
N_RECORDS = 40000
FILENAME = "dataset_revenu_marocains.csv"
RANDOM_SEED = 42 # Pour la reproductibilité
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Définition des Catégories ---
MILIEU_OPTS = ['Urbain', 'Rural']
SEXE_OPTS = ['Homme', 'Femme']
NIVEAU_EDUCATION_OPTS = ['Sans niveau', 'Fondamental', 'Secondaire', 'Supérieur']
ETAT_MATRIMONIAL_OPTS = ['Célibataire', 'Marié', 'Divorcé', 'Veuf']
CSP_OPTS = ['Cadres supérieurs', 'Professions intermédiaires', 'Employés', 'Ouvriers', 'Agriculteurs', 'Inactifs']
REGION_GEO_OPTS = ['Nord', 'Centre', 'Sud', 'Est', 'Ouest']
SECTEUR_EMPLOI_OPTS = ['Public', 'Privé', 'Informel']
OUI_NON_OPTS = ['Oui', 'Non']

# --- Proportions pour la génération (peuvent être ajustées) ---
P_URBAIN = 0.60 # Proportion cible d'urbains (ajustable)
P_RURAL = 1 - P_URBAIN

# --- Fonctions de Génération des Caractéristiques (similaires à v1, avec ajustements si besoin) ---

def generate_age(n):
    return np.random.randint(18, 64, n)

def generate_milieu(n):
    return np.random.choice(MILIEU_OPTS, n, p=[P_URBAIN, P_RURAL])

def generate_sexe(n):
    return np.random.choice(SEXE_OPTS, n, p=[0.52, 0.48])

def generate_niveau_education(n):
    return np.random.choice(NIVEAU_EDUCATION_OPTS, n, p=[0.15, 0.30, 0.35, 0.20]) # Un peu plus de 'Supérieur'

def generate_annees_experience(age, niveau_education):
    experience = []
    for a, edu in zip(age, niveau_education):
        min_age_travail = 18
        if edu == 'Supérieur': min_age_travail = 23
        elif edu == 'Secondaire': min_age_travail = 20
        
        max_exp = a - min_age_travail
        if max_exp < 0: max_exp = 0
        
        exp = np.random.randint(0, max_exp + 1) if max_exp > 0 else 0
        exp = min(exp, a - 16) if a > 16 else 0
        experience.append(max(0,exp))
    return np.array(experience)

def generate_etat_matrimonial(n, age):
    etats = []
    for a in age:
        if a < 25: etats.append(np.random.choice(ETAT_MATRIMONIAL_OPTS, p=[0.8, 0.15, 0.03, 0.02]))
        elif a < 45: etats.append(np.random.choice(ETAT_MATRIMONIAL_OPTS, p=[0.2, 0.65, 0.1, 0.05]))
        else: etats.append(np.random.choice(ETAT_MATRIMONIAL_OPTS, p=[0.1, 0.55, 0.15, 0.2]))
    return np.array(etats)

def generate_csp(niveau_education, annees_experience, age):
    csps = []
    for edu, exp, a in zip(niveau_education, annees_experience, age):
        if exp < 1 and a < 22 and edu in ['Sans niveau', 'Fondamental']: csps.append('Inactifs')
        elif edu == 'Supérieur':
            if exp > 8: csps.append('Cadres supérieurs')
            elif exp > 2: csps.append('Professions intermédiaires')
            else: csps.append(np.random.choice(['Employés', 'Professions intermédiaires'], p=[0.7,0.3]))
        elif edu == 'Secondaire':
            if exp > 12: csps.append('Professions intermédiaires')
            elif exp > 4: csps.append('Employés')
            else: csps.append(np.random.choice(['Ouvriers','Employés'], p=[0.7,0.3]))
        elif edu == 'Fondamental':
            if exp > 8: csps.append('Ouvriers')
            else: csps.append(np.random.choice(['Ouvriers', 'Agriculteurs'], p=[0.6,0.4]))
        else: # Sans niveau
            csps.append(np.random.choice(['Ouvriers', 'Agriculteurs', 'Inactifs'], p=[0.35,0.35,0.3]))
    return np.array(csps)

def generate_possession_biens(n, csp_list, milieu_list):
    prop_immo, veh_motor, terr_agri = [], [], []
    for csp, milieu in zip(csp_list, milieu_list):
        p_immo, p_veh, p_agri = 0.05, 0.05, 0.02
        if csp == 'Cadres supérieurs': p_immo, p_veh = 0.7, 0.8
        elif csp == 'Professions intermédiaires': p_immo, p_veh = 0.5, 0.6
        elif csp == 'Employés': p_immo, p_veh = 0.25, 0.35
        elif csp == 'Ouvriers': p_immo, p_veh = 0.1, 0.15
        elif csp == 'Agriculteurs': p_immo, p_veh, p_agri = 0.3, 0.25, 0.6
        
        if milieu == 'Rural':
            p_veh *= 0.7
            if csp == 'Agriculteurs': p_agri = 0.8
        
        prop_immo.append(np.random.choice(OUI_NON_OPTS, p=[p_immo, 1-p_immo]))
        veh_motor.append(np.random.choice(OUI_NON_OPTS, p=[p_veh, 1-p_veh]))
        terr_agri.append(np.random.choice(OUI_NON_OPTS, p=[p_agri, 1-p_agri]))
    return prop_immo, veh_motor, terr_agri

def generate_region_geographique(n):
    return np.random.choice(REGION_GEO_OPTS, n, p=[0.22, 0.28, 0.15, 0.15, 0.20]) # Nord, Centre, Sud, Est, Ouest

def generate_secteur_emploi(n, csp_list):
    secteurs = []
    for csp in csp_list:
        if csp == 'Inactifs': secteurs.append(np.nan)
        elif csp == 'Agriculteurs': secteurs.append(np.random.choice(['Privé', 'Informel'], p=[0.2,0.8]))
        elif csp in ['Cadres supérieurs', 'Professions intermédiaires']: secteurs.append(np.random.choice(['Public', 'Privé'], p=[0.35, 0.65]))
        elif csp == 'Employés': secteurs.append(np.random.choice(['Public', 'Privé', 'Informel'], p=[0.3, 0.5, 0.2]))
        else: secteurs.append(np.random.choice(['Privé', 'Informel'], p=[0.4, 0.6]))
    return np.array(secteurs)

def generate_revenu_secondaire(n, csp_list):
    probs = []
    for csp in csp_list:
        if csp in ['Cadres supérieurs', 'Professions intermédiaires']: probs.append(0.35)
        elif csp == 'Agriculteurs': probs.append(0.25)
        elif csp == 'Employés': probs.append(0.15)
        else: probs.append(0.05)
    return np.array([np.random.choice(OUI_NON_OPTS, p=[p, 1-p]) for p in probs])

# --- Nouvelle Fonction de Génération du Revenu Annuel ---
def adjust_group_revenus(revenus_group, target_prop, tol=0.01, max_iter=50):
    """
    Applique une transformation puissance sur les revenus du groupe afin que
    la proportion des valeurs inférieures à la moyenne soit proche de target_prop.
    Cette transformation préserve l'ordre (monotone) et devrait conserver les relations.
    """
    b_low, b_high = 0.5, 2.0
    b = 1.0
    for _ in range(max_iter):
        trans = np.power(revenus_group, b)
        new_mean = trans.mean()
        prop = (trans < new_mean).mean()
        if abs(prop - target_prop) < tol:
            return trans
        if prop > target_prop:
            b_high = b
        else:
            b_low = b
        b = (b_low + b_high) / 2.0
    return trans  # Au bout de max_iter

def generate_revenu_annuel(df):
    n = len(df)
    revenus = np.zeros(n)
    
    # Masques pour identifier les milieux
    is_urbain = (df['Milieu'] == 'Urbain').values
    is_rural = (df['Milieu'] == 'Rural').values

    # Paramètres de distribution de base distincts - Réduction des sigmas
    base_revenu_log_mean_urbain = np.log(11000)
    base_revenu_log_sigma_urbain = 0.65 # Réduit par rapport à 0.85
    base_revenu_log_mean_rural = np.log(7000)
    base_revenu_log_sigma_rural = 0.95  # Réduit par rapport à 1.15

    # Génération des revenus de base spécifiques
    if is_urbain.sum() > 0:
        revenus[is_urbain] = np.random.lognormal(mean=base_revenu_log_mean_urbain,
                                                  sigma=base_revenu_log_sigma_urbain,
                                                  size=is_urbain.sum())
    if is_rural.sum() > 0:
        revenus[is_rural] = np.random.lognormal(mean=base_revenu_log_mean_rural,
                                                 sigma=base_revenu_log_sigma_rural,
                                                 size=is_rural.sum())

    # Application des bonus/facteurs
    for i in range(n):
        record = df.iloc[i]
        rev = revenus[i]
        # Niveau d'éducation - impact augmenté
        if record['Niveau_education'] == 'Supérieur':
            rev += 25000 * (1 + 0.15 * record['Annees_experience']) # Augmentation du bonus et de l'effet de l'expérience
        elif record['Niveau_education'] == 'Secondaire':
            rev += 10000 * (1 + 0.1 * record['Annees_experience'])  # Augmentation du bonus
        elif record['Niveau_education'] == 'Fondamental':
            rev += 3000 * (1 + 0.05 * record['Annees_experience']) # Légère augmentation
        else: # Sans niveau
            rev += 750 * (1 + 0.03 * record['Annees_experience'])  # Légère augmentation

        # CSP - facteurs augmentés
        csp_factor = {
            'Cadres supérieurs': 2.5,         # Augmenté
            'Professions intermédiaires': 1.8, # Augmenté
            'Employés': 1.4,                  # Augmenté
            'Ouvriers': 1.0,                  # Ajusté
            'Agriculteurs': 0.9,              # Ajusté
            'Inactifs': 0.3                   # Réduit pour marquer la différence
        }.get(record['CSP'], 1.0)
        rev *= csp_factor

        # Sexe - impact légèrement ajusté pour plus de clarté
        rev *= 1.25 if record['Sexe'] == 'Homme' else 0.90

        # Région - impact maintenu ou légèrement ajusté si nécessaire
        region_factor = {
            'Centre': 1.1, 'Ouest': 1.1, 'Nord': 1.05,
            'Sud': 0.95, 'Est': 0.95
        }.get(record['Region_geographique'], 1.0)
        rev *= region_factor

        # Secteur Emploi - impact ajusté
        if pd.notna(record['Secteur_emploi']) and record['CSP'] != 'Inactifs':
            secteur_factor = {'Public': 1.1, 'Privé': 1.15, 'Informel': 0.85}.get(record['Secteur_emploi'], 1.0) # Ajustement des facteurs
            rev *= secteur_factor
        
        # Revenu Secondaire - impact maintenu
        if record['Revenu_secondaire'] == 'Oui':
            rev += 3000
        revenus[i] = rev

    # Calibration des moyennes par milieu
    df_temp = pd.DataFrame({'rev': revenus, 'Milieu': df['Milieu'].values})
    CIBLE_URBAIN = 26988
    CIBLE_RURAL = 12862
    mean_urbain = df_temp.loc[is_urbain, 'rev'].mean() if is_urbain.sum() > 0 else 1
    mean_rural = df_temp.loc[is_rural, 'rev'].mean() if is_rural.sum() > 0 else 1
    
    if is_urbain.sum() > 0 and mean_urbain > 0 : # Éviter la division par zéro ou NaN
        revenus[is_urbain] *= CIBLE_URBAIN / mean_urbain
    if is_rural.sum() > 0 and mean_rural > 0 : # Éviter la division par zéro ou NaN
        revenus[is_rural] *= CIBLE_RURAL / mean_rural


    # Plafonnement
    revenus = np.clip(revenus, 300, 600000)
    inactifs = (df['CSP'] == 'Inactifs').values
    revenus[inactifs] = np.minimum(revenus[inactifs], 8000)

    # Ajustement des proportions sous la moyenne
    # Vous pouvez moduler ou désactiver cette partie pour observer l'impact sur le R²
    global_target = 0.718
    urbain_target = 0.659
    rural_target = 0.854

    # S'assurer que les sous-groupes ne sont pas vides avant d'appeler adjust_group_revenus
    if revenus.size > 0 :
        revenus = adjust_group_revenus(revenus, global_target)
    if is_urbain.sum() > 0 and revenus[is_urbain].size > 0:
        revenus[is_urbain] = adjust_group_revenus(revenus[is_urbain], urbain_target)
    if is_rural.sum() > 0 and revenus[is_rural].size > 0:
        revenus[is_rural] = adjust_group_revenus(revenus[is_rural], rural_target)


    # Renormalisation des moyennes
    if is_urbain.sum() > 0 and np.mean(revenus[is_urbain]) > 0: # Éviter la division par zéro
        revenus[is_urbain] *= CIBLE_URBAIN / np.mean(revenus[is_urbain])
    if is_rural.sum() > 0 and np.mean(revenus[is_rural]) > 0: # Éviter la division par zéro
        revenus[is_rural] *= CIBLE_RURAL / np.mean(revenus[is_rural])

    # Plafonnement final
    revenus = np.clip(revenus, 300, 600000)
    return revenus

# --- Fonctions pour Imperfections (améliorées) ---
def add_valeurs_manquantes(df, colonnes_pour_nan, p_nan=0.001):
    """Ajoute des valeurs manquantes (NaN) de manière aléatoire dans les colonnes spécifiées."""
    for col in colonnes_pour_nan:
        if col in df.columns:
            mask = np.random.choice([True, False], size=len(df), p=[p_nan, 1-p_nan])
            df.loc[mask, col] = np.nan
    return df

def add_valeurs_aberrantes_age(df, p_aberrant=0.001):
    """Ajoute des valeurs aberrantes dans la colonne 'Age'."""
    n_aberr = int(p_aberrant * len(df))
    valid_indices = df.index.tolist()
    aberr_indices = random.sample(valid_indices, min(n_aberr, len(valid_indices)))
    for idx in aberr_indices:
        df.loc[idx, 'Age'] = np.random.choice([-5, 150])  # Valeurs aberrantes
    return df

def add_valeurs_aberrantes_experience(df, p_aberrant=0.001):
    """Ajoute des valeurs aberrantes dans la colonne 'Annees_experience', en tenant compte de l'âge."""
    n_aberr = int(p_aberrant * len(df))
    valid_indices = df.index.tolist()
    aberr_indices = random.sample(valid_indices, min(n_aberr, len(valid_indices)))
    for idx in aberr_indices:
        age = df.loc[idx, 'Age']
        df.loc[idx, 'Annees_experience'] = age + np.random.randint(10, 30)  # Expérience > Age
    return df

def add_valeurs_aberrantes_possession(df, p_aberrant=0.001):
    """Ajoute des valeurs aberrantes dans les colonnes de possession de biens, en tenant compte de la CSP."""
    n_aberr = int(p_aberrant * len(df))
    valid_indices = df.index.tolist()
    aberr_indices = random.sample(valid_indices, min(n_aberr, len(valid_indices)))
    for idx in aberr_indices:
        csp = df.loc[idx, 'CSP']
        if csp == 'Inactifs':
            df.loc[idx, 'Propriete_immobiliere'] = np.random.choice(['Oui', 'Non'], p=[0.9, 0.1])
            df.loc[idx, 'Vehicule_motorise'] = np.random.choice(['Oui', 'Non'], p=[0.8, 0.2])
            df.loc[idx, 'Terrain_agricole'] = np.random.choice(['Oui', 'Non'], p=[0.1, 0.9])
        elif csp == 'Agriculteurs':
            df.loc[idx, 'Terrain_agricole'] = np.random.choice(['Oui', 'Non'], p=[0.05, 0.95])
    return df

def add_valeurs_aberrantes_revenu(revenus_col, csp_col, p_aberrant=0.0005):
    """Ajoute des valeurs aberrantes dans la colonne 'Revenu_Annuel', en tenant compte de la CSP."""
    n_aberr = int(p_aberrant * len(revenus_col))
    valid_indices = revenus_col[revenus_col.notna()].index.tolist()
    aberr_indices = random.sample(valid_indices, min(n_aberr, len(valid_indices)))
    
    for idx in aberr_indices:
        csp = csp_col.loc[idx]
        if csp == 'Inactifs':
            revenus_col.loc[idx] = np.random.choice([100, 200, 300])
        elif csp == 'Cadres supérieurs':
            revenus_col.loc[idx] = np.random.choice([1000000, 1200000, 1500000])
        elif csp == 'Professions intermédiaires':
            revenus_col.loc[idx] = np.random.choice([600000, 700000])
        elif csp == 'Employés':
            revenus_col.loc[idx] = np.random.choice([400000, 500000])
        elif csp == 'Ouvriers':
            revenus_col.loc[idx] = np.random.choice([200000, 300000])
        elif csp == 'Agriculteurs':
            revenus_col.loc[idx] = np.random.choice([50000, 60000])
        else:
            revenus_col.loc[idx] = np.random.choice([300000, 400000, 500000])
    return revenus_col

# --- Génération du Dataset  ---
def generate_dataset():
    data = pd.DataFrame()

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

    data['Region_geographique'] = generate_region_geographique(N_RECORDS)
    data['Secteur_emploi'] = generate_secteur_emploi(N_RECORDS, data['CSP'])
    data['Revenu_secondaire'] = generate_revenu_secondaire(N_RECORDS, data['CSP'])
    
    data['Revenu_Annuel'] = generate_revenu_annuel(data.copy()) # Utilise la nouvelle fonction

    # Imperfections
    cols_with_nan = ['Etat_matrimonial', 'Secteur_emploi', 'Revenu_secondaire', 
                     'Propriete_immobiliere', 'Vehicule_motorise', 'Terrain_agricole',
                     'Annees_experience'] # CSP peut avoir des Inactifs, pas besoin de NaN explicite
    data = add_valeurs_manquantes(data, cols_with_nan, p_nan=0.001)
    
    # Aberrations sur le revenu (après génération et NaN)
    revenu_not_na_mask = data['Revenu_Annuel'].notna()
    data.loc[revenu_not_na_mask, 'Revenu_Annuel'] = add_valeurs_aberrantes_revenu(
        data.loc[revenu_not_na_mask, 'Revenu_Annuel'].copy(),
        data.loc[revenu_not_na_mask, 'CSP'].copy()
    )

    # AJOUT : Aberrations sur d'autres colonnes
    data = add_valeurs_aberrantes_age(data, p_aberrant=0.001)
    data = add_valeurs_aberrantes_experience(data, p_aberrant=0.001)
    data = add_valeurs_aberrantes_possession(data, p_aberrant=0.001)

    # Colonne dérivée utile pour EDA, mais pas pour modélisation directe si Age est déjà là
    bins = [18, 25, 45, 60, 100] # 100 pour couvrir les âges aberrants
    labels = ['Jeune', 'Adulte', 'Senior', 'Âgé']
    data['Categorie_age'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    
    # Vérification finale de cohérence pour Annees_experience vs Age
    mask_exp_incoherent = data['Annees_experience'] > (data['Age'] - 16)
    data.loc[mask_exp_incoherent, 'Annees_experience'] = (data.loc[mask_exp_incoherent, 'Age'] - 16 - np.random.randint(0,2, size=mask_exp_incoherent.sum())).clip(lower=0)
    
    # S'assurer que les inactifs ont bien un revenu bas ou NaN
    inactifs_mask = data['CSP'] == 'Inactifs'
    if 'Revenu_Annuel' in data.columns:
        data.loc[inactifs_mask & data['Revenu_Annuel'].notna(), 'Revenu_Annuel'] = \
            data.loc[inactifs_mask & data['Revenu_Annuel'].notna(), 'Revenu_Annuel'].apply(lambda x: min(x, 12000))

    # AJOUT: Colonnes Redondantes et Non Pertinentes
    data['Revenu_Mensuel'] = (data['Revenu_Annuel'] / 12).round(2)
    data['Adresse_Email'] = [f"user{uuid.uuid4().hex[:6]}@example.com" for _ in range(N_RECORDS)]
    data['CIN'] = [f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000,999999)}" for _ in range(N_RECORDS)]

    ordered_columns = [
        'Age', 'Categorie_age', 'Sexe', 'Milieu', 'Region_geographique', 'Etat_matrimonial',
        'Niveau_education', 'Annees_experience', 'CSP', 'Secteur_emploi',
        'Propriete_immobiliere', 'Vehicule_motorise', 'Terrain_agricole',
        'Revenu_secondaire', 'Revenu_Annuel', 'Revenu_Mensuel',  # Ajout de Revenu_Mensuel
        'Adresse_Email', 'CIN'  # Ajout de Adresse_Email et CIN
    ]
    final_columns = [col for col in ordered_columns if col in data.columns]
    missing_cols = [col for col in data.columns if col not in final_columns] # Au cas où
    data = data[final_columns + missing_cols]

    return data

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Génération de {N_RECORDS} enregistrements avec le script ...")
    dataset = generate_dataset()
    
    print("\n--- Statistiques Descriptives du Revenu Annuel  ---")
    print(dataset['Revenu_Annuel'].describe())
    
    # --- Vérification des Contraintes HCP ---
    print("\n--- Vérification des Contraintes Statistiques (Basées sur HCP) ---")
    
    # Revenus moyens globaux et par milieu
    revenu_moyen_global_genere = dataset['Revenu_Annuel'].mean()
    print(f"Revenu Annuel Moyen Global (Généré): {revenu_moyen_global_genere:.0f} DH (Cible HCP: 21949 DH)")

    moyennes_milieu_genere = dataset.groupby('Milieu')['Revenu_Annuel'].mean()
    revenu_moyen_urbain_genere = moyennes_milieu_genere.get('Urbain', np.nan)
    revenu_moyen_rural_genere = moyennes_milieu_genere.get('Rural', np.nan)

    print(f"Revenu Annuel Moyen Urbain (Généré): {revenu_moyen_urbain_genere:.0f} DH (Cible HCP: 26988 DH)")
    print(f"Revenu Annuel Moyen Rural (Généré): {revenu_moyen_rural_genere:.0f} DH (Cible HCP: 12862 DH)")

    # Proportions de revenus inférieurs à la moyenne
    if pd.notna(revenu_moyen_global_genere):
        prop_inf_moy_global = (dataset['Revenu_Annuel'] < revenu_moyen_global_genere).mean() * 100
        print(f"Proportion revenus < moyenne globale (Généré): {prop_inf_moy_global:.1f}% (Cible HCP: 71.8%)")
    else:
        print("Proportion revenus < moyenne globale (Généré): Non calculable (moyenne globale NaN)")

    if pd.notna(revenu_moyen_urbain_genere) and 'Urbain' in moyennes_milieu_genere:
        prop_inf_moy_urbain = (dataset[dataset['Milieu'] == 'Urbain']['Revenu_Annuel'] < revenu_moyen_urbain_genere).mean() * 100
        print(f"Proportion revenus < moyenne urbaine (Généré): {prop_inf_moy_urbain:.1f}% (Cible HCP: 65.9%)")
    else:
        print("Proportion revenus < moyenne urbaine (Généré): Non calculable")
    
    if pd.notna(revenu_moyen_rural_genere) and 'Rural' in moyennes_milieu_genere:
        prop_inf_moy_rural = (dataset[dataset['Milieu'] == 'Rural']['Revenu_Annuel'] < revenu_moyen_rural_genere).mean() * 100
        print(f"Proportion revenus < moyenne rurale (Généré): {prop_inf_moy_rural:.1f}% (Cible HCP: 85.4%)")
    else:
        print("Proportion revenus < moyenne rurale (Généré): Non calculable")

    print("\nMoyennes par CSP (Généré) :")
    print(dataset.groupby('CSP')['Revenu_Annuel'].mean().sort_values(ascending=False))

    print("\nMoyennes par Niveau d'éducation (Généré) :")
    print(dataset.groupby('Niveau_education')['Revenu_Annuel'].mean().sort_values(ascending=False))

    dataset.to_csv(FILENAME, index=False, encoding='utf-8')
    print(f"\nDataset '{FILENAME}' généré avec succès ({len(dataset)} enregistrements).")
