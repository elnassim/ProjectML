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
# - Revenu moyen urbain: 26,988 DH
# - Revenu moyen rural: 12,862 DH

# Pour atteindre ces moyennes et la répartition urbain/rural correcte, nous devons ajuster:
# 1. La proportion urbain/rural (calculée pour atteindre la moyenne globale cible)
# 2. Les paramètres des distributions lognormales pour les revenus urbains et ruraux

# Calcul de la proportion urbain/rural pour atteindre la moyenne globale cible
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
    # Urbain/rural selon la proportion calculée
    "area": np.random.choice(["urbain", "rural"], num_samples, p=[urban_proportion, rural_proportion]),
    
    # Âge: distribution normale entre 18 et 70 ans
    "age": np.clip(np.random.normal(35, 15, num_samples).astype(int), 18, 70),
    
    # Genre: 55% hommes, 45% femmes
    "gender": np.random.choice(["homme", "femme"], num_samples, p=[0.55, 0.45]),
    
    # Niveau d'éducation
    "education": np.random.choice(
        ["sans_niveau", "fondamental", "secondaire", "supérieur"], 
        num_samples, 
        p=[0.2, 0.3, 0.3, 0.2]
    ),
    
    # Groupe socio-professionnel
    "socio_professional_group": np.random.choice(
        ["Groupe1", "Groupe2", "Groupe3", "Groupe4", "Groupe5", "Groupe6"], 
        num_samples, 
        p=[0.1, 0.2, 0.15, 0.15, 0.25, 0.15]
    ),
    
    # Années d'expérience: corrélées avec l'âge
    "years_experience": np.zeros(num_samples),
    
    # État civil
    "marital_status": np.random.choice(
        ["célibataire", "marié", "divorcé", "veuf"], 
        num_samples, 
        p=[0.4, 0.4, 0.1, 0.1]
    )
}

# Années d'expérience corrélées avec l'âge (entre 0 et âge-18)
for i in range(num_samples):
    max_exp = max(0, data["age"][i] - 18)
    exp_mean = max_exp * 0.7  # En moyenne, 70% du temps depuis 18 ans
    data["years_experience"][i] = min(max_exp, max(0, np.random.normal(exp_mean, 3)))
data["years_experience"] = data["years_experience"].astype(int)

# ==============================================
# 3. Génération des caractéristiques supplémentaires
# ==============================================
# Définir urban_mask
urban_mask = (data["area"] == "urbain")

# 1. Voiture
data["has_car"] = np.where(urban_mask, 
                          np.random.choice([0, 1], num_samples, p=[0.3, 0.7]),
                          np.random.choice([0, 1], num_samples, p=[0.6, 0.4]))

# 2. Logement
data["home_ownership"] = np.where(urban_mask,
    np.random.choice(["owned", "rented", "other"], num_samples, p=[0.4, 0.5, 0.1]),
    np.random.choice(["owned", "rented", "other"], num_samples, p=[0.7, 0.2, 0.1]))

# 3. Terrain
data["has_land"] = np.where(data["area"] == "rural", 
                          np.random.choice([0, 1], num_samples, p=[0.2, 0.8]), 
                          np.random.choice([0, 1], num_samples, p=[0.9, 0.1]))

# 4. Nombre d'enfants (corrélé avec l'état civil)
data["number_of_children"] = np.zeros(num_samples, dtype=int)
married_mask = (data["marital_status"] == "marié")
data["number_of_children"][married_mask] = np.clip(np.random.poisson(1.5, married_mask.sum()), 0, 5)
data["number_of_children"][~married_mask] = np.clip(np.random.poisson(0.7, (~married_mask).sum()), 0, 3)

# 5. Source de revenu secondaire
df_temp = pd.DataFrame(data)
data["source_revenu_secondaire"] = np.where(
    df_temp["area"] == "rural",
    # En rural : 60% ont une source secondaire (agriculture, élevage)
    np.random.choice([0, 1], num_samples, p=[0.4, 0.6]), 
    # En urbain : 35% ont une source secondaire (commerce, location)
    np.random.choice([0, 1], num_samples, p=[0.65, 0.35])  
)

# 6. Assurance maladie
data["health_insurance"] = np.where(
    df_temp["socio_professional_group"].isin(["Groupe1", "Groupe2"]),
    np.random.choice([0, 1], num_samples, p=[0.2, 0.8]),
    np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
)

# ==============================================
# 4. Génération des revenus
# ==============================================
# Base du revenu (distribution lognormale)
income = np.zeros(num_samples)

# Générer les revenus de base selon la zone
urban_mask = (data["area"] == "urbain")
rural_mask = ~urban_mask

income[urban_mask] = np.random.lognormal(urban_mu, urban_sigma, urban_mask.sum())
income[rural_mask] = np.random.lognormal(rural_mu, rural_sigma, rural_mask.sum())

# Appliquer des multiplicateurs pour les différents facteurs
multipliers = np.ones(num_samples)

# 4.a – Effet Urbain vs Rural
# les urbains gagnent en moyenne 15% de plus
multipliers *= np.where(data["area"]=="urbain", 1.15, 1.00)

# Éducation
edu_multipliers = {
    "sans_niveau": 0.85,
    "fondamental": 0.95,
    "secondaire": 1.1,
    "supérieur": 1.4
}
edu_mult = np.array([edu_multipliers.get(data["education"][i], 1.0) for i in range(num_samples)])
multipliers *= edu_mult

# Groupe socio‑professionnel
group_multipliers = {
    "Groupe1": 1.6, "Groupe2": 1.3, "Groupe3": 1.1,
    "Groupe4": 0.9, "Groupe5": 0.8, "Groupe6": 0.7
}
group_mult = np.array([group_multipliers.get(data["socio_professional_group"][i],1.0)
                       for i in range(num_samples)])
multipliers *= group_mult

# Sexe: hommes +15%, femmes –15%
multipliers *= np.where(np.array(data["gender"])=="homme", 1.15, 0.85)

# Expérience (fonction quadratique)
exp_years = np.array(data["years_experience"])
exp_effect = 0.8 + 0.01 * exp_years + 0.0005 * (exp_years**2)
multipliers *= np.clip(exp_effect, 0.8, 2.0)

# Âge (cloche autour de 52‑55 ans)
age_arr = np.array(data["age"])
age_effect = 0.7 + 0.02 * age_arr - 0.0002 * (age_arr**2)
multipliers *= np.clip(age_effect, 0.7, 1.3)

# 4.b – Effet Catégorie d’âge
# jeune, adulte, sénior, âgé
df = pd.DataFrame(data)  # Ensure df is defined before use
df['age_category'] = pd.cut(df['age'], bins=[18, 26, 46, 61, 71], labels=['jeune', 'adulte', 'sénior', 'âgé'], right=False)

cat_mult = df['age_category'].map({
    'jeune': 0.90,
    'adulte': 1.00,
    'sénior': 1.10,
    'âgé':    1.05
}).fillna(1.0).values.astype(float)
multipliers *= cat_mult

# Appliquer bruit et arrondi final
final_income = income * multipliers
noise = np.random.lognormal(0, 0.2, num_samples)
final_income *= noise
data["income"] = np.round(final_income).astype(int)

# ==============================================
# 5. Ajustement pour obtenir les cibles exactes
# ==============================================
df = pd.DataFrame(data)

# Vérifier les moyennes actuelles
current_mean = df['income'].mean()
current_urban_mean = df[df['area'] == 'urbain']['income'].mean()
current_rural_mean = df[df['area'] == 'rural']['income'].mean()

print(f"Moyennes initiales:")
print(f"Global: {current_mean:.2f} DH (cible: 21949)")
print(f"Urbain: {current_urban_mean:.2f} DH (cible: 26988)")
print(f"Rural: {current_rural_mean:.2f} DH (cible: 12862)")

# Ajustement pour atteindre les moyennes cibles
urban_factor = 26988 / current_urban_mean
rural_factor = 12862 / current_rural_mean

# Appliquer les facteurs d'ajustement
df.loc[df['area'] == 'urbain', 'income'] = (df.loc[df['area'] == 'urbain', 'income'] * urban_factor).round().astype(int)
df.loc[df['area'] == 'rural', 'income'] = (df.loc[df['area'] == 'rural', 'income'] * rural_factor).round().astype(int)

# Vérifier les moyennes ajustées
adjusted_mean = df['income'].mean()
adjusted_urban_mean = df[df['area'] == 'urbain']['income'].mean()
adjusted_rural_mean = df[df['area'] == 'rural']['income'].mean()

print(f"\nMoyennes ajustées:")
print(f"Global: {adjusted_mean:.2f} DH (cible: 21949)")
print(f"Urbain: {adjusted_urban_mean:.2f} DH (cible: 26988)")
print(f"Rural: {adjusted_rural_mean:.2f} DH (cible: 12862)")

# Vérifier les pourcentages en dessous de la moyenne
pct_below_mean = (df['income'] < adjusted_mean).mean() * 100
pct_below_urban_mean = (df[df['area'] == 'urbain']['income'] < adjusted_urban_mean).mean() * 100
pct_below_rural_mean = (df[df['area'] == 'rural']['income'] < adjusted_rural_mean).mean() * 100

print(f"\nPourcentages en dessous de la moyenne:")
print(f"Global: {pct_below_mean:.1f}% (cible: 71.8%)")
print(f"Urbain: {pct_below_urban_mean:.1f}% (cible: 65.9%)")
print(f"Rural: {pct_below_rural_mean:.1f}% (cible: 85.4%)")
# ==============================================
# 6. Ajustement itératif des % < moyenne
# ==============================================

def adjust_distribution_for_target_percentile(incomes, mean_value, target_percent_below):
    """
    Ajuste un vecteur d'incomes pour obtenir target_percent_below % de valeurs < mean_value.
    """
    incomes = incomes.copy().astype(float)
    for _ in range(10):
        pct_below = (incomes < mean_value).mean() * 100
        error = target_percent_below - pct_below
        if abs(error) < 0.05:
            break
        # si on a trop peu de valeurs < mean, on diminue légèrement tous les revenus
        # sinon on augmente légèrement
        factor = 1 - error/1000
        incomes = incomes * factor
    return incomes

# cibles
target_pct = {'global': 71.8, 'urbain': 65.9, 'rural': 85.4}

mask_u = df['area']=='urbain'
mask_r = df['area']=='rural'

# itération sur global/rural/urbain
for _ in range(5):
    # recalcul des moyennes
    m_g = df['income'].mean()
    m_u = df.loc[mask_u,'income'].mean()
    m_r = df.loc[mask_r,'income'].mean()

    # ajustement urbain
    df.loc[mask_u, 'income'] = adjust_distribution_for_target_percentile(
        df.loc[mask_u,'income'].values, m_u, target_pct['urbain']
    ).round().astype(int)
    # ajustement rural
    df.loc[mask_r, 'income'] = adjust_distribution_for_target_percentile(
        df.loc[mask_r,'income'].values, m_r, target_pct['rural']
    ).round().astype(int)
    # ajustement global
    df['income'] = adjust_distribution_for_target_percentile(
        df['income'].values, m_g, target_pct['global']
    ).round().astype(int)

# vérification finale
final_global_mean = df['income'].mean()
final_urban_mean  = df.loc[mask_u,'income'].mean()
final_rural_mean  = df.loc[mask_r,'income'].mean()

final_pct_global = (df['income'] < final_global_mean).mean()*100
final_pct_urbain = (df.loc[mask_u,'income'] < final_urban_mean).mean()*100
final_pct_rural  = (df.loc[mask_r,'income'] < final_rural_mean).mean()*100

print(f"Répartition des revenus : {final_pct_global:.1f}% < moyenne (urbain {final_pct_urbain:.1f}%, rural {final_pct_rural:.1f}%)")


# ==============================================
# 7. Ajout de la catégorie d'âge
# ==============================================
bins = [18, 26, 46, 61, 71]
labels = ['jeune', 'adulte', 'sénior', 'âgé']
df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# ==============================================
# 8. Ajout de problèmes de qualité des données
# ==============================================
# Valeurs manquantes (5-10% dans diverses colonnes)
missing_mask = np.random.rand(len(df)) < 0.07
df.loc[missing_mask, "education"] = np.nan

df["years_experience"] = df["years_experience"].mask(np.random.rand(len(df)) < 0.05, np.nan)
df["home_ownership"] = df["home_ownership"].mask(np.random.rand(len(df)) < 0.03, np.nan)

# Outliers (1% élevés, 0.5% bas)
high_outliers = np.random.choice(len(df), int(len(df)*0.01), replace=False)
df.loc[high_outliers, "income"] = np.random.randint(100000, 500000, len(high_outliers))

low_outliers = np.random.choice(len(df), int(len(df)*0.005), replace=False)
df.loc[low_outliers, "income"] = np.random.randint(0, 1000, len(low_outliers))

# Colonnes redondantes ou non pertinentes
df["year_of_birth"] = 2024 - df["age"]                    # Redondante
df["age_squared"] = df["age"] ** 2                        # Redondante
df["favorite_color"] = np.random.choice(                  # Non pertinente
    ["rouge", "bleu", "vert", "jaune"], len(df))

# ==============================================
# 9. Calibration finale des revenus et vérifications
# ==============================================

# Calibration des moyennes finales
targets = {'urbain': 26988, 'rural': 12862}
global_tgt = 21949

# Calibration par zone
for area, tgt in targets.items():
    mask = df['area'] == area
    current_mean = df.loc[mask, 'income'].mean()
    df.loc[mask, 'income'] = (df.loc[mask, 'income'] * (tgt / current_mean)).round().astype(int)

# Calibration de la moyenne globale
current_global_mean = df['income'].mean()
df['income'] = (df['income'] * (global_tgt / current_global_mean)).round().astype(int)

# Calcul des moyennes finales
final_global_mean = df['income'].mean()
final_urban_mean = df[df['area'] == 'urbain']['income'].mean()
final_rural_mean = df[df['area'] == 'rural']['income'].mean()

# Calcul des pourcentages sous la moyenne
final_pct_below_global = (df['income'] < final_global_mean).mean() * 100
final_pct_below_urban = (df[df['area'] == 'urbain']['income'] < final_urban_mean).mean() * 100
final_pct_below_rural = (df[df['area'] == 'rural']['income'] < final_rural_mean).mean() * 100

# Affichage des vérifications finales
print("\n----- Vérifications finales -----")
print(f"Moyenne globale : {final_global_mean:.2f} DH (cible : 21 949)")
print(f"Moyenne urbaine : {final_urban_mean:.2f} DH (cible : 26 988)")
print(f"Moyenne rurale  : {final_rural_mean:.2f} DH (cible : 12 862)")

print(f"\nRépartition des revenus : {final_pct_below_global:.1f}% < moyenne (urbain {final_pct_below_urban:.1f}%, rural {final_pct_below_rural:.1f}%)")
print("Cibles : 71,8% (urbain 65,9%, rural 85,4%)")
df['income'] = (df['income'] * (global_tgt/current_global_mean)).round().astype(int)
# ==============================================
# 10. Sauvegarde du dataset
# ==============================================
df.to_csv("dataset_revenu_marocains.csv", index=False)
print("\nDataset généré avec succès !")