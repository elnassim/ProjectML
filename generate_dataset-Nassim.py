import pandas as pd
import numpy as np
from scipy import stats

# Pour reproductibilité
np.random.seed(42)
num_samples = 40000

# ==============================================
# 1. Paramètres pour atteindre les moyennes cibles
# ==============================================
# Cibles:
# - Revenu moyen global: 21,949 DH
# - Revenu moyen Urbain: 26,988 DH
# - Revenu moyen Rural: 12,862 DH

# Pour atteindre ces moyennes et la répartition Urbain/Rural correcte, nous devons ajuster:
# 1. La proportion Urbain/Rural (calculée pour atteindre la moyenne globale cible)
# 2. Les paramètres des distributions lognormales pour les revenus urbains et ruraux

# Calcul de la proportion Urbain/Rural pour atteindre la moyenne globale cible
# Si x = proportion urbaine, alors:
# x * 26988 + (1-x) * 12862 = 21949
# Résolution: x = (21949 - 12862) / (26988 - 12862) = 0.643

urban_proportion = 0.643  # Proportion urbaine calculée pour atteindre la moyenne globale
rural_proportion = 1 - urban_proportion

# Paramètres de distribution lognormale pour obtenir les moyennes cibles
# Pour une distribution lognormale, la moyenne = exp(mu + sigma²/2)
# On résout pour mu: mu = ln(moyenne) - sigma²/2

# Urbain
urban_sigma = 0.55  # Écart-type choisi pour avoir une distribution réaliste
urban_mu = np.log(26988) - (urban_sigma**2)/2

# Rural
rural_sigma = 0.85  # Écart-type choisi pour avoir une distribution réaliste
rural_mu = np.log(12862) - (rural_sigma**2)/2

# ==============================================
# 2. Génération des caractéristiques de base
# ==============================================
data = {
    # Urbain/Rural selon la proportion calculée
    "Milieu": np.random.choice(["Urbain", "Rural"], num_samples, p=[urban_proportion, rural_proportion]),
    
    # Âge: distribution normale entre 18 et 70 ans
    "Age": np.clip(np.random.normal(40, 12, num_samples).astype(int), 16, 66),
    
    # Genre: 55% hommes, 45% femmes
    "Sexe": np.random.choice(["Homme", "Femme"], num_samples, p=[0.55, 0.45]),
    
    # Niveau d'éducation
    "Niveau_education": np.random.choice(
        ["Sans niveau", "Fondamental", "Secondaire", "Supérieur"], 
        num_samples, 
        p=[0.2, 0.3, 0.3, 0.2]
    ),
    
    # Groupe socio-professional group
    "CSP": np.random.choice(
        ["Cadres supérieurs", "Professions intermédiaires", "Employés", "Ouvriers", "Agriculteurs", "Inactifs"], 
        num_samples, 
        p=[0.1, 0.2, 0.15, 0.15, 0.25, 0.15]
    ),
    
    # Années d'expérience: corrélées avec l'âge
    "Annees_experience": np.zeros(num_samples),
    
    # État civil
    "Etat_matrimonial": np.random.choice(
        ["Célibataire", "Marié", "Divorcé", "Veuf"], 
        num_samples, 
        p=[0.4, 0.4, 0.1, 0.1]
    ),

    "Region_geographique": np.random.choice(
        ["Nord", "Centre", "Ouest", "Sud", "Est"],
        num_samples,
        p=[0.20, 0.30, 0.25, 0.15, 0.10]  # Exemple de probabilités
    )

    
}

# Années d'expérience corrélées avec l'âge (entre 0 et âge-18)
for i in range(num_samples):
    max_exp = max(0, data["Age"][i] - 16)
    exp_mean = max_exp * 0.7  # En moyenne, 70% du temps depuis 18 ans
    data["Annees_experience"][i] = min(max_exp, max(0, np.random.normal(exp_mean, 3)))
data["Annees_experience"] = data["Annees_experience"].astype(int)

# ==============================================
# 3. Génération des caractéristiques supplémentaires
# ==============================================
# Définir urban_mask
urban_mask = (data["Milieu"] == "Urbain")

# 1. Voiture
data["Vehicule_motorise"] = np.where(urban_mask, 
                          np.random.choice([0, 1], num_samples, p=[0.3, 0.7]),
                          np.random.choice([0, 1], num_samples, p=[0.6, 0.4]))

data["Propriete_immobiliere"] = np.where(urban_mask,
    np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),  # Urban: 40% own homes
    np.random.choice([0, 1], num_samples, p=[0.3, 0.7])) 

# 3. Terrain
data["Terrain_agricole"] = np.where(data["Milieu"] == "Rural", 
                          np.random.choice([0, 1], num_samples, p=[0.2, 0.8]), 
                          np.random.choice([0, 1], num_samples, p=[0.9, 0.1]))


# 5. Source de revenu Secondaire
df_temp = pd.DataFrame(data)
data["Revenu_secondaire"] = np.where(
    df_temp["Milieu"] == "Rural",
    # En Rural : 60% ont une source Secondaire (agriculture, élevage)
    np.random.choice([0, 1], num_samples, p=[0.4, 0.6]), 
    # En Urbain : 35% ont une source Secondaire (commerce, location)
    np.random.choice([0, 1], num_samples, p=[0.65, 0.35])  
)

# 6. Secteur d'emploi (NEW)
# Urban: 30% Public, 50% Privé, 20% Informel
# Rural: 10% Public, 20% Privé, 70% Informel
data["Secteur_emploi"] = np.where(
    urban_mask,
    np.random.choice(
        ["Privé ", "Public", "Informel"], 
        num_samples, 
        p=[0.5, 0.3, 0.2]  # Probabilities for Urban: Privé 50%, Public 30%, Informel 20%
    ),
    np.random.choice(
        ["Privé ", "Public", "Informel"],
        num_samples,
        p=[0.2, 0.1, 0.7]  # Probabilities for Rural: Privé 20%, Public 10%, Informel 70%
    )
)


# ==============================================
# 4. Génération des revenus
# ==============================================

# Initialize multipliers as an array of ones
multipliers = np.ones(num_samples)

multipliers *= np.where(data["Milieu"]=="Urbain", 1.15, 1.00)

# Nouveau: Effet Secteur d'emploi (NEW)
secteur_multipliers = {
    "Public": 1.15,  # Increased from 0.95 to 1.15
    "Privé ": 2.0,
    "Informel": 0.7
}
secteur_mult = np.array([secteur_multipliers[s] for s in data["Secteur_emploi"]])
multipliers *= secteur_mult

# Éducation (existing code)
edu_multipliers = {
    "Sans niveau": 0.85,
    "Fondamental": 0.95,
    "Secondaire": 1.1,
    "Supérieur": 1.4
}

# Base du revenu (distribution lognormale)
Revenu_Annuel = np.zeros(num_samples)

# Générer les revenus de base selon la zone
urban_mask = (data["Milieu"] == "Urbain")
rural_mask = ~urban_mask

Revenu_Annuel[urban_mask] = np.random.lognormal(urban_mu, urban_sigma, urban_mask.sum())
Revenu_Annuel[rural_mask] = np.random.lognormal(rural_mu, rural_sigma, rural_mask.sum())

# Appliquer des multiplicateurs pour les différents facteurs
multipliers = np.ones(num_samples)

# 4.a – Effet Urbain vs Rural
# les urbains gagnent en moyenne 15% de plus
multipliers *= np.where(data["Milieu"]=="Urbain", 1.15, 1.00)
multipliers *= secteur_mult # Add this line

# Éducation
edu_multipliers = {
    "Sans niveau": 0.85,
    "Fondamental": 0.95,
    "Secondaire": 1.1,
    "Supérieur": 1.4
}
edu_mult = np.array([edu_multipliers.get(data["Niveau_education"][i], 1.0) for i in range(num_samples)])
multipliers *= edu_mult

# Groupe socio‑professionnel
group_multipliers = {
    "Cadres supérieurs": 2.0,  # Increased
    "Professions intermédiaires": 1.6,  # Increased
    "Employés": 1.3,  # Increased
    "Ouvriers": 1.0,  # Adjusted
    "Agriculteurs": 0.8,  # Kept or slightly adjusted
    "Inactifs": 0.6   # Decreased
}
group_mult = np.array([group_multipliers.get(data["CSP"][i],1.0)
                       for i in range(num_samples)])
multipliers *= group_mult


region_multipliers = {
    "Nord": 1.15,    # Pôle économique (ex: Tanger)
    "Centre": 1.25,  # Pôle économique majeur (ex: Casablanca-Rabat)
    "Ouest": 1.20,   # Pôle économique (ex: autres régions côtières actives)
    "Sud": 0.85,     # Moins élevé
    "Est": 0.80      # Moins élevé
}

region_mult_array = np.array([region_multipliers.get(data["Region_geographique"][i], 1.0)
                           for i in range(num_samples)])
multipliers *= region_mult_array
# Sexe: hommes +15%, femmes –15%
multipliers *= np.where(np.array(data["Sexe"])=="Homme", 1.15, 0.85)

# Expérience (fonction quadratique)
exp_years = np.array(data["Annees_experience"])
exp_effect = 0.8 + 0.01 * exp_years + 0.0005 * (exp_years**2)
multipliers *= np.clip(exp_effect, 0.8, 2.0)

# Âge (cloche autour de 52‑55 ans)
age_arr = np.array(data["Age"])
age_effect = 0.636736 + 0.010204 * age_arr
multipliers *= age_effect

# 4.b – Effet Catégorie d'âge
# Jeune, Adulte, Senior, âgé
df = pd.DataFrame(data)  # Ensure df is defined before use
df['Categorie_age'] = pd.cut(df['Age'], bins=[16, 26, 46, 61, 66], labels=['Jeune', 'Adulte', 'Senior', 'âgé'], right=False)

cat_mult = df['Categorie_age'].map({
    'Jeune': 0.90,
    'Adulte': 1.00,
    'Senior': 1.10,
    'Âgé':    1.05
}).fillna(1.0).values.astype(float)
multipliers *= cat_mult


# Effet Source de Revenu Secondaire (NEW)
# Ceux avec un revenu Secondaire gagnent, par exemple, 10% de plus
secondary_income_multiplier = 1.1 
multipliers *= np.where(np.array(data["Revenu_secondaire"]) == 1, secondary_income_multiplier, 1.0)


# Multiplicateurs selon l'état matrimonial et le genre
marital_status_multipliers = {
    'Homme': {
        'Célibataire': 0.85,   # -15% pour les hommes célibataires (base réduite)
        'Marié': 1.15,         # +15% pour les hommes mariés
        'Divorcé': 1.12,       # +12% pour les hommes divorcés
        'Veuf': 1.10           # +10% pour les hommes veufs
    },
    'Femme': {
        'Célibataire': 0.85,   # -15% pour les femmes célibataires (base réduite)
        'Marié': 1.05,         # +5% pour les femmes mariées
        'Divorcé': 1.40,       # +40% pour les femmes divorcées
        'Veuf': 1.35           # +35% pour les femmes veuves
    }
}

# Appliquer les multiplicateurs d'état matrimonial
marital_mult = np.array([
    marital_status_multipliers[data["Sexe"][i]][data["Etat_matrimonial"][i]]
    for i in range(num_samples)
])
multipliers *= marital_mult

# Appliquer bruit et arrondi final
final_income = Revenu_Annuel * multipliers
noise = np.random.lognormal(0, 0.2, num_samples)
final_income *= noise
data["Revenu_Annuel"] = np.round(final_income).astype(int)

# ==============================================
# 5. Ajustement pour obtenir les cibles exactes
# ==============================================
df = pd.DataFrame(data)

# Ajustement basé sur les possessions
def adjust_income_by_possessions(df):
   
    
    # Calculer le nombre total de possessions pour chaque personne
    df['total_possessions'] = df['Vehicule_motorise'] + df['Propriete_immobiliere'] + df['Terrain_agricole']
    
    # Calculer les revenus moyens par nombre de possessions
    mean_income_by_possessions = df.groupby('total_possessions')['Revenu_Annuel'].mean()
    
    # Définir les multiplicateurs pour chaque niveau de possessions
    possession_multipliers = {
        0: 0.85,  # -15% pour ceux sans possessions
        1: 1.05,  # +5% pour ceux avec 1 possession
        2: 1.15,  # +15% pour ceux avec 2 possessions
        3: 1.30   # +30% pour ceux avec 3 possessions
    }
    
    # Appliquer les ajustements
    for num_possessions, multiplier in possession_multipliers.items():
        mask = df['total_possessions'] == num_possessions
        if mask.any():
            current_mean = df.loc[mask, 'Revenu_Annuel'].mean()
            target_mean = mean_income_by_possessions.mean() * multiplier
            adjustment_factor = target_mean / current_mean
            df.loc[mask, 'Revenu_Annuel'] = (df.loc[mask, 'Revenu_Annuel'] * adjustment_factor).round().astype(int)
    
    return df

# Appliquer l'ajustement des possessions
df = adjust_income_by_possessions(df)

# Vérification des revenus moyens par nombre de possessions
print("\n----- Vérification des revenus moyens par nombre de possessions -----")
income_by_possessions = df.groupby('total_possessions')['Revenu_Annuel'].agg(['mean', 'count'])
print("\nRevenu moyen par nombre de possessions:")
print(income_by_possessions)

# Vérification des différences de revenu
print("\nDifférences de revenu entre les niveaux de possessions:")
for i in range(3):
    current_mean = income_by_possessions.loc[i, 'mean']
    next_mean = income_by_possessions.loc[i+1, 'mean']
    diff_percent = ((next_mean / current_mean) - 1) * 100
    print(f"{i+1} possession(s) vs {i} possession(s): {diff_percent:.1f}% plus élevé")

# Vérification des revenus moyens par type de possession
print("\nRevenu moyen par type de possession:")
print("\nVoiture:")
print(df.groupby('Vehicule_motorise')['Revenu_Annuel'].mean())
# REPLACE with verification by Propriete_immobiliere:
print("\nPropriété immobilière (binaire):")
print(df.groupby('Propriete_immobiliere')['Revenu_Annuel'].mean())
print("\nTerrain:")
print(df.groupby('Terrain_agricole')['Revenu_Annuel'].mean())

# Ajustement post-traitement en deux phases
def adjust_marital_income(df):
    # Phase 1: Ajuster les revenus des célibataires pour les deux genres
    for Sexe in ['Homme', 'Femme']:
        # Calculer la moyenne des revenus non-célibataires
        non_single_mask = (df['Sexe'] == Sexe) & (df['Etat_matrimonial'] != 'Célibataire')
        non_single_mean = df[non_single_mask]['Revenu_Annuel'].mean()
        
        # Ajuster les revenus des célibataires
        single_mask = (df['Sexe'] == Sexe) & (df['Etat_matrimonial'] == 'Célibataire')
        df.loc[single_mask, 'Revenu_Annuel'] = (df.loc[single_mask, 'Revenu_Annuel'] * 
                                       (0.85 * non_single_mean / df.loc[single_mask, 'Revenu_Annuel'].mean()))
    
    # Phase 2: Ajuster spécifiquement les revenus des femmes divorcées et veuves
    # Calculer les revenus moyens de référence pour les femmes
    single_women_mean = df[(df['Sexe'] == 'Femme') & (df['Etat_matrimonial'] == 'Célibataire')]['Revenu_Annuel'].mean()
    married_women_mean = df[(df['Sexe'] == 'Femme') & (df['Etat_matrimonial'] == 'Marié')]['Revenu_Annuel'].mean()
    reference_mean = max(single_women_mean, married_women_mean)
    
    # Ajuster les revenus des femmes divorcées et veuves
    divorced_mask = (df['Sexe'] == 'Femme') & (df['Etat_matrimonial'] == 'Divorcé')
    widowed_mask = (df['Sexe'] == 'Femme') & (df['Etat_matrimonial'] == 'Veuf')
    
    # Calculer les facteurs d'ajustement
    divorced_factor = 1.40  # +40% par rapport à la référence
    widowed_factor = 1.35   # +35% par rapport à la référence
    
    # Appliquer les ajustements
    df.loc[divorced_mask, 'Revenu_Annuel'] = (df.loc[divorced_mask, 'Revenu_Annuel'] * 
                                     (divorced_factor * reference_mean / df.loc[divorced_mask, 'Revenu_Annuel'].mean()))
    df.loc[widowed_mask, 'Revenu_Annuel'] = (df.loc[widowed_mask, 'Revenu_Annuel'] * 
                                    (widowed_factor * reference_mean / df.loc[widowed_mask, 'Revenu_Annuel'].mean()))
    
    return df

# Appliquer l'ajustement post-traitement
df = adjust_marital_income(df)

# Vérifier les moyennes actuelles
current_mean = df['Revenu_Annuel'].mean()
current_urban_mean = df[df['Milieu'] == 'Urbain']['Revenu_Annuel'].mean()
current_rural_mean = df[df['Milieu'] == 'Rural']['Revenu_Annuel'].mean()

print(f"Moyennes initiales:")
print(f"Global: {current_mean:.2f} DH (cible: 21949)")
print(f"Urbain: {current_urban_mean:.2f} DH (cible: 26988)")
print(f"Rural: {current_rural_mean:.2f} DH (cible: 12862)")

# Ajustement pour atteindre les moyennes cibles
urban_factor = 26988 / current_urban_mean
rural_factor = 12862 / current_rural_mean

# Appliquer les facteurs d'ajustement
df.loc[df['Milieu'] == 'Urbain', 'Revenu_Annuel'] = (df.loc[df['Milieu'] == 'Urbain', 'Revenu_Annuel'] * urban_factor).round().astype(int)
df.loc[df['Milieu'] == 'Rural', 'Revenu_Annuel'] = (df.loc[df['Milieu'] == 'Rural', 'Revenu_Annuel'] * rural_factor).round().astype(int)

# Vérification des revenus moyens par genre et état matrimonial
print("\n----- Vérification des revenus moyens par genre et état matrimonial -----")
income_by_gender_marital = df.groupby(['Sexe', 'Etat_matrimonial'])['Revenu_Annuel'].mean()
print("\nRevenu moyen par genre et état matrimonial:")
print(income_by_gender_marital)

# Vérification spécifique pour les femmes
women_income = income_by_gender_marital.loc['Femme']
print("\nRevenu moyen des femmes par état matrimonial:")
print(women_income)

# Vérification que les femmes divorcées et veuves gagnent plus que les célibataires et mariées
single_women_income = women_income['Célibataire']
married_women_income = women_income['Marié']
divorced_women_income = women_income['Divorcé']
widowed_women_income = women_income['Veuf']

print(f"\nDifférence de revenu pour les femmes:")
print(f"Divorcées vs Célibataires: {((divorced_women_income/single_women_income - 1) * 100):.1f}%")
print(f"Divorcées vs Mariées: {((divorced_women_income/married_women_income - 1) * 100):.1f}%")
print(f"Veuves vs Célibataires: {((widowed_women_income/single_women_income - 1) * 100):.1f}%")
print(f"Veuves vs Mariées: {((widowed_women_income/married_women_income - 1) * 100):.1f}%")

# Vérification pour les hommes
men_income = income_by_gender_marital.loc['Homme']
print("\nRevenu moyen des hommes par état matrimonial:")
print(men_income)

# Vérification que les hommes célibataires gagnent moins que les autres
single_men_income = men_income['Célibataire']
print(f"\nDifférence de revenu pour les hommes:")
print(f"Mariés vs Célibataires: {((men_income['Marié']/single_men_income - 1) * 100):.1f}%")
print(f"Divorcés vs Célibataires: {((men_income['Divorcé']/single_men_income - 1) * 100):.1f}%")
print(f"Veufs vs Célibataires: {((men_income['Veuf']/single_men_income - 1) * 100):.1f}%")

# ==============================================
# 6. Ajustement itératif des % < moyenne
# ==============================================

def adjust_distribution_for_target_percentile(incomes, mean_value, target_percent_below, area_type='other', phase=1):
    """
    Ajuste un vecteur d'incomes pour obtenir target_percent_below % de valeurs < mean_value.
    Phase 1: Ajustement initial
    Phase 2: Ajustement fin
    Phase 3: Ajustement ultra-fin
    Phase 4: Ajustement micro
    Phase 5: Ajustement nano
    Phase 6: Ajustement pico
    Phase 7: Ajustement femto
    """
    incomes = incomes.copy().astype(float)
    
    if area_type == 'urban':
        # Stratégie en sept phases pour Urbain
        for _ in range(30):
            pct_below = (incomes < mean_value).mean() * 100
            if phase == 1:
                if pct_below <= 67.0:
                    break
                factor = 1.015  # 1.5% d'augmentation
            elif phase == 2:
                if pct_below <= 66.2:
                    break
                factor = 1.003  # 0.3% d'augmentation
            elif phase == 3:
                if pct_below <= 66.0:
                    break
                factor = 1.001  # 0.1% d'augmentation
            elif phase == 4:
                if pct_below <= 65.95:
                    break
                factor = 1.0003  # 0.03% d'augmentation
            elif phase == 5:
                if pct_below <= 65.92:
                    break
                factor = 1.0001  # 0.01% d'augmentation
            elif phase == 6:
                if pct_below <= 65.91:
                    break
                factor = 1.00005  # 0.005% d'augmentation
            else:  # phase 7
                if pct_below <= 65.9:
                    break
                # Ajustement femto
                if pct_below > 65.91:
                    factor = 1.00002  # 0.002% d'augmentation
                else:
                    factor = 1.000005  # 0.0005% d'augmentation
            incomes = incomes * factor
            
    elif area_type == 'Rural':
        # Stratégie en sept phases pour Rural
        for _ in range(30):
            pct_below = (incomes < mean_value).mean() * 100
            if phase == 1:
                if pct_below >= 84.0:
                    break
                factor = 0.985  # 1.5% de réduction
            elif phase == 2:
                if pct_below >= 85.0:
                    break
                factor = 0.997  # 0.3% de réduction
            elif phase == 3:
                if pct_below >= 85.2:
                    break
                factor = 0.999  # 0.1% de réduction
            elif phase == 4:
                if pct_below >= 85.35:
                    break
                factor = 0.9997  # 0.03% de réduction
            elif phase == 5:
                if pct_below >= 85.38:
                    break
                factor = 0.9999  # 0.01% de réduction
            elif phase == 6:
                if pct_below >= 85.41:
                    break
                factor = 0.99995  # 0.005% de réduction
            else:  # phase 7
                if pct_below >= 85.4:
                    break
                # Ajustement femto
                if pct_below < 85.41:
                    factor = 0.99998  # 0.002% de réduction
                else:
                    factor = 0.999995  # 0.0005% de réduction
            incomes = incomes * factor
            
    else:  # global
        # Ajustement ultra-conservateur pour le global
        for _ in range(25):
            pct_below = (incomes < mean_value).mean() * 100
            error = target_percent_below - pct_below
            if abs(error) < 0.002:  # Tolérance femto
                break
            factor = 1 - (error/500000)  # Encore plus conservateur
            factor = np.clip(factor, 0.9999, 1.0001)  # Limite femto
            incomes = incomes * factor
            
    return incomes

# Cibles exactes
target_pct = {
    'global': 71.8,
    'Urbain': 65.9,
    'Rural': 85.4
}

mask_u = df['Milieu']=='Urbain'
mask_r = df['Milieu']=='Rural'

# Phase 1: Ajustement initial
for _ in range(2):
    m_g = df['Revenu_Annuel'].mean()
    m_u = df.loc[mask_u,'Revenu_Annuel'].mean()
    m_r = df.loc[mask_r,'Revenu_Annuel'].mean()

    # Ajustements phase 1
    df.loc[mask_u, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'Revenu_Annuel'].values, m_u, target_pct['Urbain'], area_type='urban', phase=1
    ).round().astype(int)
    
    df.loc[mask_r, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'Revenu_Annuel'].values, m_r, target_pct['Rural'], area_type='Rural', phase=1
    ).round().astype(int)
    
    df['Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df['Revenu_Annuel'].values, m_g, target_pct['global'], area_type='global'
    ).round().astype(int)

# Phase 2: Ajustement fin
for _ in range(2):
    m_g = df['Revenu_Annuel'].mean()
    m_u = df.loc[mask_u,'Revenu_Annuel'].mean()
    m_r = df.loc[mask_r,'Revenu_Annuel'].mean()

    # Ajustements phase 2
    df.loc[mask_u, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'Revenu_Annuel'].values, m_u, target_pct['Urbain'], area_type='urban', phase=2
    ).round().astype(int)
    
    df.loc[mask_r, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'Revenu_Annuel'].values, m_r, target_pct['Rural'], area_type='Rural', phase=2
    ).round().astype(int)
    
    df['Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df['Revenu_Annuel'].values, m_g, target_pct['global'], area_type='global'
    ).round().astype(int)

# Phase 3: Ajustement ultra-fin
for _ in range(3):
    m_g = df['Revenu_Annuel'].mean()
    m_u = df.loc[mask_u,'Revenu_Annuel'].mean()
    m_r = df.loc[mask_r,'Revenu_Annuel'].mean()

    # Ajustements phase 3
    df.loc[mask_u, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'Revenu_Annuel'].values, m_u, target_pct['Urbain'], area_type='urban', phase=3
    ).round().astype(int)
    
    df.loc[mask_r, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'Revenu_Annuel'].values, m_r, target_pct['Rural'], area_type='Rural', phase=3
    ).round().astype(int)
    
    df['Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df['Revenu_Annuel'].values, m_g, target_pct['global'], area_type='global'
    ).round().astype(int)

# Phase 4: Ajustement micro
for _ in range(4):
    m_g = df['Revenu_Annuel'].mean()
    m_u = df.loc[mask_u,'Revenu_Annuel'].mean()
    m_r = df.loc[mask_r,'Revenu_Annuel'].mean()

    # Ajustements phase 4
    df.loc[mask_u, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'Revenu_Annuel'].values, m_u, target_pct['Urbain'], area_type='urban', phase=4
    ).round().astype(int)
    
    df.loc[mask_r, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'Revenu_Annuel'].values, m_r, target_pct['Rural'], area_type='Rural', phase=4
    ).round().astype(int)
    
    df['Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df['Revenu_Annuel'].values, m_g, target_pct['global'], area_type='global'
    ).round().astype(int)

# Phase 5: Ajustement nano
for _ in range(5):
    m_g = df['Revenu_Annuel'].mean()
    m_u = df.loc[mask_u,'Revenu_Annuel'].mean()
    m_r = df.loc[mask_r,'Revenu_Annuel'].mean()

    # Ajustements phase 5
    df.loc[mask_u, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'Revenu_Annuel'].values, m_u, target_pct['Urbain'], area_type='urban', phase=5
    ).round().astype(int)
    
    df.loc[mask_r, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'Revenu_Annuel'].values, m_r, target_pct['Rural'], area_type='Rural', phase=5
    ).round().astype(int)
    
    df['Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df['Revenu_Annuel'].values, m_g, target_pct['global'], area_type='global'
    ).round().astype(int)

# Phase 6: Ajustement pico
for _ in range(6):
    m_g = df['Revenu_Annuel'].mean()
    m_u = df.loc[mask_u,'Revenu_Annuel'].mean()
    m_r = df.loc[mask_r,'Revenu_Annuel'].mean()

    # Ajustements phase 6
    df.loc[mask_u, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'Revenu_Annuel'].values, m_u, target_pct['Urbain'], area_type='urban', phase=6
    ).round().astype(int)
    
    df.loc[mask_r, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'Revenu_Annuel'].values, m_r, target_pct['Rural'], area_type='Rural', phase=6
    ).round().astype(int)
    
    df['Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df['Revenu_Annuel'].values, m_g, target_pct['global'], area_type='global'
    ).round().astype(int)

# Phase 7: Ajustement femto
for _ in range(7):
    m_g = df['Revenu_Annuel'].mean()
    m_u = df.loc[mask_u,'Revenu_Annuel'].mean()
    m_r = df.loc[mask_r,'Revenu_Annuel'].mean()

    # Ajustements phase 7
    df.loc[mask_u, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'Revenu_Annuel'].values, m_u, target_pct['Urbain'], area_type='urban', phase=7
    ).round().astype(int)
    
    df.loc[mask_r, 'Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'Revenu_Annuel'].values, m_r, target_pct['Rural'], area_type='Rural', phase=7
    ).round().astype(int)
    
    df['Revenu_Annuel'] = adjust_distribution_for_target_percentile(
        df['Revenu_Annuel'].values, m_g, target_pct['global'], area_type='global'
    ).round().astype(int)
    
    # Vérification des pourcentages actuels
    current_pct_urban = (df.loc[mask_u, 'Revenu_Annuel'] < m_u).mean() * 100
    current_pct_rural = (df.loc[mask_r, 'Revenu_Annuel'] < m_r).mean() * 100
    
    # Arrêt si on est femto-proche des cibles
    if (abs(current_pct_urban - target_pct['Urbain']) < 0.005 and 
        abs(current_pct_rural - target_pct['Rural']) < 0.005):
        break
        
    # Arrêt d'urgence si on dépasse les limites
    if current_pct_urban < 65.89 or current_pct_rural > 85.41:
        break

# Vérification finale
final_global_mean = df['Revenu_Annuel'].mean()
final_urban_mean = df.loc[mask_u, 'Revenu_Annuel'].mean()
final_rural_mean = df.loc[mask_r, 'Revenu_Annuel'].mean()

final_pct_global = (df['Revenu_Annuel'] < final_global_mean).mean()*100
final_pct_urbain = (df.loc[mask_u, 'Revenu_Annuel'] < final_urban_mean).mean()*100
final_pct_rural = (df.loc[mask_r, 'Revenu_Annuel'] < final_rural_mean).mean()*100

print("\n----- Vérification des pourcentages sous la moyenne -----")
print(f"Global: {final_pct_global:.1f}% < moyenne (cible: {target_pct['global']}%)")
print(f"Urbain: {final_pct_urbain:.1f}% < moyenne (cible: {target_pct['Urbain']}%)")
print(f"Rural: {final_pct_rural:.1f}% < moyenne (cible: {target_pct['Rural']}%)")

# ==============================================
# 7. Ajout de la catégorie d'âge
# ==============================================
bins = [16, 26, 46, 61, 66]
labels = ['Jeune', 'Adulte', 'Senior', 'Âgé']
df['Categorie_age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# ==============================================
# 8. Ajout de problèmes de qualité des données
# ==============================================
# Valeurs manquantes (5-10% dans diverses colonnes)
missing_mask = np.random.rand(len(df)) < 0.07
df.loc[missing_mask, "Niveau_education"] = np.nan

df["Annees_experience"] = df["Annees_experience"].mask(np.random.rand(len(df)) < 0.05, np.nan)
df["Propriete_immobiliere"] = df["Propriete_immobiliere"].mask(np.random.rand(len(df)) < 0.03, np.nan)

# Outliers (1% élevés, 0.5% bas)
high_outliers = np.random.choice(len(df), int(len(df)*0.01), replace=False)
df.loc[high_outliers, "Revenu_Annuel"] = np.random.randint(100000, 500000, len(high_outliers))

low_outliers = np.random.choice(len(df), int(len(df)*0.005), replace=False)
df.loc[low_outliers, "Revenu_Annuel"] = np.random.randint(0, 1000, len(low_outliers))

# Colonnes redondantes ou non pertinentes
  
df["Revenu_Mensuel"] = (df["Revenu_Annuel"] / 12).round(2) # Ajout de la colonne redondante Revenu_Mensuel                 

# Ajout d'une colonne Adresse_Email non pertinente
import string
surnames = ["alami", "benjelloun", "cherkaoui", "drissi", "elalaoui", "fassi", "guerrouj", "haddadi", "idrissi", "jouahri"]
first_names = ["ahmed", "fatima", "mohamed", "khadija", "youssef", "amina", "omar", "sara", "ali", "layla"]
domains = ["fakemail.com", "example.org", "testmail.net", "anonymous.io"]

emails = []
for _ in range(len(df)):
    name_part = np.random.choice(first_names) + "." + np.random.choice(surnames)
    domain_part = np.random.choice(domains)
    random_suffix = ''.join(np.random.choice(list(string.digits), np.random.randint(0, 4))) # Optional: add some random numbers
    emails.append(f"{name_part}{random_suffix}@{domain_part}")
df["Adresse_Email"] = emails

# Ajout d'une colonne CIN non pertinente
cin_list = []
for _ in range(len(df)):
    # Generate a random CIN-like string (e.g., 2 letters + 6 digits)
    letters = ''.join(np.random.choice(list(string.ascii_uppercase), 2))
    numbers = ''.join(np.random.choice(list(string.digits), 6))
    cin_list.append(f"{letters}{numbers}")
df["CIN"] = cin_list
# ==============================================
# 9. Calibration finale des revenus et vérifications
# ==============================================

print("\n----- Début de la Calibration Finale des Revenus -----")

# Définition des cibles exactes
global_target_mean = 21949.0
urban_target_mean = 26988.0
rural_target_mean = 12862.0

# Créer des masques pour Urbain et Rural si non existants
mask_u = (df['Milieu'] == 'Urbain')
mask_r = (df['Milieu'] == 'Rural')

# 1. Ajustement des moyennes urbaines et rurales
current_urban_mean = df.loc[mask_u, 'Revenu_Annuel'].mean()
current_rural_mean = df.loc[mask_r, 'Revenu_Annuel'].mean()

print(f"Moyenne urbaine AVANT calibration spécifique: {current_urban_mean:,.2f} DH")
print(f"Moyenne rurale AVANT calibration spécifique: {current_rural_mean:,.2f} DH")

if pd.notna(current_urban_mean) and current_urban_mean > 0:
    urban_adjustment_factor = urban_target_mean / current_urban_mean
    df.loc[mask_u, 'Revenu_Annuel'] = df.loc[mask_u, 'Revenu_Annuel'] * urban_adjustment_factor
    print(f"Facteur d'ajustement Urbain appliqué: {urban_adjustment_factor:.4f}")
else:
    print("Avertissement: Moyenne urbaine actuelle est NaN ou nulle. Ajustement Urbain sauté.")

if pd.notna(current_rural_mean) and current_rural_mean > 0:
    rural_adjustment_factor = rural_target_mean / current_rural_mean
    df.loc[mask_r, 'Revenu_Annuel'] = df.loc[mask_r, 'Revenu_Annuel'] * rural_adjustment_factor
    print(f"Facteur d'ajustement Rural appliqué: {rural_adjustment_factor:.4f}")
else:
    print("Avertissement: Moyenne rurale actuelle est NaN ou nulle. Ajustement Rural sauté.")

# Vérification des moyennes après ajustement Urbain/Rural
# print(f"Moyenne urbaine APRÈS calibration spécifique: {df.loc[mask_u, 'Revenu_Annuel'].mean():,.2f} DH")
# print(f"Moyenne rurale APRÈS calibration spécifique: {df.loc[mask_r, 'Revenu_Annuel'].mean():,.2f} DH")

# 2. Ajustement de la moyenne globale
# La moyenne globale peut avoir changé après les ajustements spécifiques Urbain/Rural
current_global_mean_after_ur_adj = df['Revenu_Annuel'].mean()
print(f"Moyenne globale AVANT calibration globale finale: {current_global_mean_after_ur_adj:,.2f} DH")

if pd.notna(current_global_mean_after_ur_adj) and current_global_mean_after_ur_adj > 0:
    global_adjustment_factor = global_target_mean / current_global_mean_after_ur_adj
    df['Revenu_Annuel'] = df['Revenu_Annuel'] * global_adjustment_factor
    print(f"Facteur d'ajustement global final appliqué: {global_adjustment_factor:.4f}")
else:
    print("Avertissement: Moyenne globale actuelle est NaN ou nulle. Ajustement global final sauté.")

# Arrondir les revenus et s'assurer d'un minimum (pour éviter des revenus négatifs ou trop bas)
df['Revenu_Annuel'] = df['Revenu_Annuel'].round().astype(int)
df['Revenu_Annuel'] = np.maximum(df['Revenu_Annuel'], 100) # Assurer un revenu minimum (ex: 100 DH)

# --- Vérifications finales après TOUTE calibration ---
final_global_mean = df['Revenu_Annuel'].mean()
final_urban_mean  = df.loc[mask_u,'Revenu_Annuel'].mean()
final_rural_mean  = df.loc[mask_r,'Revenu_Annuel'].mean()

print("\n----- Vérifications finales (après calibration ciblée) -----")
print(f"Moyenne globale finale: {final_global_mean:,.2f} DH (Cible: {global_target_mean:,.2f} DH)")
print(f"Moyenne urbaine finale: {final_urban_mean:,.2f} DH (Cible: {urban_target_mean:,.2f} DH)")
print(f"Moyenne rurale finale : {final_rural_mean:,.2f} DH (Cible: {rural_target_mean:,.2f} DH)")

final_pct_below_global = (df['Revenu_Annuel'] < final_global_mean).mean() * 100
final_pct_below_urban = (df.loc[mask_u, 'Revenu_Annuel'] < final_urban_mean).mean() * 100
final_pct_below_rural = (df.loc[mask_r, 'Revenu_Annuel'] < final_rural_mean).mean() * 100
print(f"\nRépartition des revenus : {final_pct_below_global:.1f}% < moyenne (Urbain {final_pct_below_urban:.1f}%, Rural {final_pct_below_rural:.1f}%)")
print("Cibles : 71,8% (Urbain 65,9%, Rural 85,4%)")
current_global_mean = df['Revenu_Annuel'].mean()  # Calculate the global mean
global_tgt = 21949  # Define the global target mean Revenu_Annuel
df['Revenu_Annuel'] = (df['Revenu_Annuel'] * (global_tgt/current_global_mean)).round().astype(int)

print("\nMoyennes par secteur d'emploi:")
print(df.groupby("Secteur_emploi")["Revenu_Annuel"].mean().sort_values(ascending=False))


# --- BEGIN NEW BLOCK: Final Adjustment for Secondary Revenu_Annuel Effect ---
print("\n----- Ajustement final pour l'effet du revenu Secondaire -----")
mean_income_sec_yes_before = df[df['Revenu_secondaire'] == 1]['Revenu_Annuel'].mean()
mean_income_sec_no_before = df[df['Revenu_secondaire'] == 0]['Revenu_Annuel'].mean()

print(f"Moyenne AVANT ajustement (Revenu Secondaire Oui): {mean_income_sec_yes_before:,.0f} DH")
print(f"Moyenne AVANT ajustement (Revenu Secondaire Non): {mean_income_sec_no_before:,.0f} DH")

# Define how much higher the 'yes' group's mean Revenu_Annuel should be (e.g., 5% higher)
desired_mean_ratio = 1.05 

if pd.notna(mean_income_sec_yes_before) and pd.notna(mean_income_sec_no_before) and \
   (mean_income_sec_yes_before <= mean_income_sec_no_before or \
    (mean_income_sec_yes_before / mean_income_sec_no_before) < desired_mean_ratio if mean_income_sec_no_before > 0 else False):
    
    print(f"Ajustement en cours pour que le groupe 'Revenu Secondaire Oui' ait un revenu moyen ~{((desired_mean_ratio-1)*100):.0f}% plus élevé...")
    
    if mean_income_sec_yes_before > 0 and mean_income_sec_no_before > 0:
        # Calculate the factor needed to apply to the 'yes' group
        target_mean_yes = mean_income_sec_no_before * desired_mean_ratio
        adjustment_factor = target_mean_yes / mean_income_sec_yes_before
        
        # Apply adjustment only to the 'yes' group
        df.loc[df['Revenu_secondaire'] == 1, 'Revenu_Annuel'] = \
            (df.loc[df['Revenu_secondaire'] == 1, 'Revenu_Annuel'] * adjustment_factor).round().astype(int)
        
        # Ensure incomes don't go below a certain minimum (e.g., 0 or a small positive value)
        df.loc[df['Revenu_secondaire'] == 1, 'Revenu_Annuel'] = df.loc[df['Revenu_secondaire'] == 1, 'Revenu_Annuel'].apply(lambda x: max(x, 100))


        mean_income_sec_yes_after = df[df['Revenu_secondaire'] == 1]['Revenu_Annuel'].mean()
        mean_income_sec_no_after = df[df['Revenu_secondaire'] == 0]['Revenu_Annuel'].mean() # Should be same as before this specific adjustment
        
        print(f"Moyenne APRÈS ajustement (Revenu Secondaire Oui): {mean_income_sec_yes_after:,.0f} DH")
        print(f"Moyenne APRÈS ajustement (Revenu Secondaire Non): {mean_income_sec_no_after:,.0f} DH")
        if mean_income_sec_no_after > 0:
             print(f"Ratio Oui/Non après ajustement: {(mean_income_sec_yes_after/mean_income_sec_no_after):.2f} (Cible: {desired_mean_ratio:.2f})")
    else:
        print("Ajustement non effectué (une des moyennes est zéro ou négative).")
else:
    print("Aucun ajustement nécessaire, la relation de revenu souhaitée pour le revenu Secondaire est déjà respectée ou les données sont insuffisantes.")

# Note: This adjustment might slightly alter the overall global/urban/Rural means.
# You can re-calculate and print them here if needed for final verification.
final_global_mean_after_all_adj = df['Revenu_Annuel'].mean()
print(f"\nMoyenne globale finale (après TOUS les ajustements): {final_global_mean_after_all_adj:,.2f} DH")
# --- END NEW BLOCK ---
# ==============================================
# 10. Calibration finale des revenus moyens
# ==============================================
print("\n----- Calibration finale des revenus moyens -----")

# Cibles des revenus moyens
target_means = {
    'global': 21949.0,
    'Urbain': 26988.0,
    'Rural': 12862.0
}

# Calcul des moyennes actuelles
current_global_mean = df['Revenu_Annuel'].mean()
current_urban_mean = df.loc[mask_u, 'Revenu_Annuel'].mean()
current_rural_mean = df.loc[mask_r, 'Revenu_Annuel'].mean()

print(f"Moyennes AVANT calibration finale:")
print(f"Global: {current_global_mean:,.2f} DH (cible: {target_means['global']:,.2f} DH)")
print(f"Urbain: {current_urban_mean:,.2f} DH (cible: {target_means['Urbain']:,.2f} DH)")
print(f"Rural: {current_rural_mean:,.2f} DH (cible: {target_means['Rural']:,.2f} DH)")

# Calcul des facteurs d'ajustement
global_factor = target_means['global'] / current_global_mean
urban_factor = target_means['Urbain'] / current_urban_mean
rural_factor = target_means['Rural'] / current_rural_mean

# Application des ajustements en préservant les distributions
df.loc[mask_u, 'Revenu_Annuel'] = (df.loc[mask_u, 'Revenu_Annuel'] * urban_factor).round().astype(int)
df.loc[mask_r, 'Revenu_Annuel'] = (df.loc[mask_r, 'Revenu_Annuel'] * rural_factor).round().astype(int)

# Vérification des pourcentages après ajustement
final_global_mean = df['Revenu_Annuel'].mean()
final_urban_mean = df.loc[mask_u, 'Revenu_Annuel'].mean()
final_rural_mean = df.loc[mask_r, 'Revenu_Annuel'].mean()

final_pct_global = (df['Revenu_Annuel'] < final_global_mean).mean() * 100
final_pct_urban = (df.loc[mask_u, 'Revenu_Annuel'] < final_urban_mean).mean() * 100
final_pct_rural = (df.loc[mask_r, 'Revenu_Annuel'] < final_rural_mean).mean() * 100

print(f"\nMoyennes APRÈS calibration finale:")
print(f"Global: {final_global_mean:,.2f} DH (cible: {target_means['global']:,.2f} DH)")
print(f"Urbain: {final_urban_mean:,.2f} DH (cible: {target_means['Urbain']:,.2f} DH)")
print(f"Rural: {final_rural_mean:,.2f} DH (cible: {target_means['Rural']:,.2f} DH)")

print(f"\nPourcentages sous la moyenne:")
print(f"Global: {final_pct_global:.1f}% (cible: 71.8%)")
print(f"Urbain: {final_pct_urban:.1f}% (cible: 65.9%)")
print(f"Rural: {final_pct_rural:.1f}% (cible: 85.4%)")

# Sauvegarde finale
df_export = df.drop(columns=['total_possessions'])

# Réorganiser les colonnes dans l'ordre spécifié
columns_order = [
    'Age', 'Categorie_age', 'Sexe', 'Milieu', 'Region_geographique', 
    'Etat_matrimonial', 'Niveau_education', 'Annees_experience', 
    'CSP', 'Secteur_emploi', 'Propriete_immobiliere', 'Vehicule_motorise', 
    'Terrain_agricole', 'Revenu_secondaire', 'Revenu_Annuel', 
    'Revenu_Mensuel', 'Adresse_Email', 'CIN'
]

# Réorganiser et sauvegarder
df_export = df_export[columns_order]

df_export.to_csv("dataset_revenu_marocains.csv", index=False)
print("\nDataset généré avec succès !")