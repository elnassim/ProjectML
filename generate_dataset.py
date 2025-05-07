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
    "age": np.clip(np.random.normal(40, 12, num_samples).astype(int), 16, 75),
    
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
    ),

    "region_geographique": np.random.choice(
        ["Nord", "Centre", "Ouest", "Sud", "Est"],
        num_samples,
        p=[0.20, 0.30, 0.25, 0.15, 0.10]  # Exemple de probabilités
    )

    
}

# Années d'expérience corrélées avec l'âge (entre 0 et âge-18)
for i in range(num_samples):
    max_exp = max(0, data["age"][i] - 16)
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


# 5. Source de revenu secondaire
df_temp = pd.DataFrame(data)
data["source_revenu_secondaire"] = np.where(
    df_temp["area"] == "rural",
    # En rural : 60% ont une source secondaire (agriculture, élevage)
    np.random.choice([0, 1], num_samples, p=[0.4, 0.6]), 
    # En urbain : 35% ont une source secondaire (commerce, location)
    np.random.choice([0, 1], num_samples, p=[0.65, 0.35])  
)

# 6. Secteur d'emploi (NEW)
# Urban: 30% Public, 50% Privé, 20% Informel
# Rural: 10% Public, 20% Privé, 70% Informel
data["secteur_emploi"] = np.where(
    urban_mask,
    np.random.choice(
        ["Privé formel", "Public", "Informel"], 
        num_samples, 
        p=[0.5, 0.3, 0.2]  # Probabilities for Urban: Privé 50%, Public 30%, Informel 20%
    ),
    np.random.choice(
        ["Privé formel", "Public", "Informel"],
        num_samples,
        p=[0.2, 0.1, 0.7]  # Probabilities for Rural: Privé 20%, Public 10%, Informel 70%
    )
)


# ==============================================
# 4. Génération des revenus
# ==============================================

# Initialize multipliers as an array of ones
multipliers = np.ones(num_samples)

multipliers *= np.where(data["area"]=="urbain", 1.15, 1.00)

# Nouveau: Effet Secteur d'emploi (NEW)
secteur_multipliers = {
    "Public": 0.95,
    "Privé formel": 2.0,
    "Informel": 0.7
}
secteur_mult = np.array([secteur_multipliers[s] for s in data["secteur_emploi"]])
multipliers *= secteur_mult

# Éducation (existing code)
edu_multipliers = {
    "sans_niveau": 0.85,
    "fondamental": 0.95,
    "secondaire": 1.1,
    "supérieur": 1.4
}

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
multipliers *= secteur_mult # Add this line

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
    "Groupe1": 2.0,  # Increased
    "Groupe2": 1.6,  # Increased
    "Groupe3": 1.3,  # Increased
    "Groupe4": 1.0,  # Adjusted
    "Groupe5": 0.8,  # Kept or slightly adjusted
    "Groupe6": 0.6   # Decreased
}
group_mult = np.array([group_multipliers.get(data["socio_professional_group"][i],1.0)
                       for i in range(num_samples)])
multipliers *= group_mult


region_multipliers = {
    "Nord": 1.15,    # Pôle économique (ex: Tanger)
    "Centre": 1.25,  # Pôle économique majeur (ex: Casablanca-Rabat)
    "Ouest": 1.20,   # Pôle économique (ex: autres régions côtières actives)
    "Sud": 0.85,     # Moins élevé
    "Est": 0.80      # Moins élevé
}

region_mult_array = np.array([region_multipliers.get(data["region_geographique"][i], 1.0)
                           for i in range(num_samples)])
multipliers *= region_mult_array
# Sexe: hommes +15%, femmes –15%
gender_array = np.array(data["gender"])
marital_status_array = np.array(data["marital_status"])

# Initialize a new multiplier array for marital status and gender combined
# Default to 1.0, will be overridden by specific conditions
marital_gender_effect_multipliers = np.ones(num_samples)

# Define multipliers for each combination based on the constraints:
# Hommes: mariés, divorcés, veufs > célibataires
# Femmes: divorcées, veuves > célibataires/mariées

# Multiplicateurs pour les hommes
marital_gender_effect_multipliers[
    (gender_array == 'homme') & (marital_status_array == 'célibataire')
] = 1.10  # Base pour hommes célibataires
marital_gender_effect_multipliers[
    (gender_array == 'homme') & (marital_status_array == 'marié')
] = 1.20  # Plus élevé
marital_gender_effect_multipliers[
    (gender_array == 'homme') & (marital_status_array == 'divorcé')
] = 1.18  # Plus élevé
marital_gender_effect_multipliers[
    (gender_array == 'homme') & (marital_status_array == 'veuf')
] = 1.17  # Plus élevé

# Multiplicateurs pour les femmes
marital_gender_effect_multipliers[
    (gender_array == 'femme') & (marital_status_array == 'célibataire')
] = 0.85  # Base pour femmes célibataires
marital_gender_effect_multipliers[
    (gender_array == 'femme') & (marital_status_array == 'marié')
] = 0.85  # Base pour femmes mariées (similaire à célibataire selon l'interprétation du besoin)
marital_gender_effect_multipliers[
    (gender_array == 'femme') & (marital_status_array == 'divorcé')
] = 0.95  # Plus élevé
marital_gender_effect_multipliers[
    (gender_array == 'femme') & (marital_status_array == 'veuf')
] = 0.92  # Plus élevé

# Appliquer ce nouveau multiplicateur combiné
multipliers *= marital_gender_effect_multipliers

# Expérience (fonction quadratique)
exp_years = np.array(data["years_experience"])
exp_effect = 0.8 + 0.01 * exp_years + 0.0005 * (exp_years**2)
multipliers *= np.clip(exp_effect, 0.8, 2.0)

# Âge (cloche autour de 52‑55 ans)
age_arr = np.array(data["age"])
age_effect = 0.7 + 0.02 * age_arr - 0.0002 * (age_arr**2)
multipliers *= np.clip(age_effect, 0.7, 1.3)

# 4.b – Effet Catégorie d'âge
# jeune, adulte, sénior, âgé
df = pd.DataFrame(data)  # Ensure df is defined before use
df['age_category'] = pd.cut(df['age'], bins=[18, 26, 46, 61, 71], labels=['jeune', 'adulte', 'sénior', 'âgé'], right=False)

cat_mult = df['age_category'].map({
    'jeune': 0.90,
    'adulte': 1.00,
    'sénior': 1.10,
    'agé':    1.05
}).fillna(1.0).values.astype(float)
multipliers *= cat_mult


# Effet Source de Revenu Secondaire (NEW)
# Ceux avec un revenu secondaire gagnent, par exemple, 10% de plus
secondary_income_multiplier = 1.1 
multipliers *= np.where(np.array(data["source_revenu_secondaire"]) == 1, secondary_income_multiplier, 1.0)


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
bins = [16, 26, 46, 61, 66]
labels = ['jeune', 'adulte', 'sénior', 'agé']
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
df["year_of_birth"] = 2025 - df["age"]     
df["revenue_mensuel"] = (df["income"] / 12).round(2) # Ajout de la colonne redondante revenue_mensuel                 
df["favorite_color"] = np.random.choice(                  # Non pertinente
    ["rouge", "bleu", "vert", "jaune"], len(df))
# Ajout d'une colonne email non pertinente
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
df["email"] = emails

# Ajout d'une colonne CIN non pertinente
cin_list = []
for _ in range(len(df)):
    # Generate a random CIN-like string (e.g., 2 letters + 6 digits)
    letters = ''.join(np.random.choice(list(string.ascii_uppercase), 1))
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

# Créer des masques pour urbain et rural si non existants
mask_u = (df['area'] == 'urbain')
mask_r = (df['area'] == 'rural')

# 1. Ajustement des moyennes urbaines et rurales
current_urban_mean = df.loc[mask_u, 'income'].mean()
current_rural_mean = df.loc[mask_r, 'income'].mean()

print(f"Moyenne urbaine AVANT calibration spécifique: {current_urban_mean:,.2f} DH")
print(f"Moyenne rurale AVANT calibration spécifique: {current_rural_mean:,.2f} DH")

if pd.notna(current_urban_mean) and current_urban_mean > 0:
    urban_adjustment_factor = urban_target_mean / current_urban_mean
    df.loc[mask_u, 'income'] = df.loc[mask_u, 'income'] * urban_adjustment_factor
    print(f"Facteur d'ajustement urbain appliqué: {urban_adjustment_factor:.4f}")
else:
    print("Avertissement: Moyenne urbaine actuelle est NaN ou nulle. Ajustement urbain sauté.")

if pd.notna(current_rural_mean) and current_rural_mean > 0:
    rural_adjustment_factor = rural_target_mean / current_rural_mean
    df.loc[mask_r, 'income'] = df.loc[mask_r, 'income'] * rural_adjustment_factor
    print(f"Facteur d'ajustement rural appliqué: {rural_adjustment_factor:.4f}")
else:
    print("Avertissement: Moyenne rurale actuelle est NaN ou nulle. Ajustement rural sauté.")

# Vérification des moyennes après ajustement urbain/rural
# print(f"Moyenne urbaine APRÈS calibration spécifique: {df.loc[mask_u, 'income'].mean():,.2f} DH")
# print(f"Moyenne rurale APRÈS calibration spécifique: {df.loc[mask_r, 'income'].mean():,.2f} DH")

# 2. Ajustement de la moyenne globale
# La moyenne globale peut avoir changé après les ajustements spécifiques urbain/rural
current_global_mean_after_ur_adj = df['income'].mean()
print(f"Moyenne globale AVANT calibration globale finale: {current_global_mean_after_ur_adj:,.2f} DH")

if pd.notna(current_global_mean_after_ur_adj) and current_global_mean_after_ur_adj > 0:
    global_adjustment_factor = global_target_mean / current_global_mean_after_ur_adj
    df['income'] = df['income'] * global_adjustment_factor
    print(f"Facteur d'ajustement global final appliqué: {global_adjustment_factor:.4f}")
else:
    print("Avertissement: Moyenne globale actuelle est NaN ou nulle. Ajustement global final sauté.")

# Arrondir les revenus et s'assurer d'un minimum (pour éviter des revenus négatifs ou trop bas)
df['income'] = df['income'].round().astype(int)
df['income'] = np.maximum(df['income'], 100) # Assurer un revenu minimum (ex: 100 DH)

# --- Vérifications finales après TOUTE calibration ---
final_global_mean = df['income'].mean()
final_urban_mean  = df.loc[mask_u,'income'].mean()
final_rural_mean  = df.loc[mask_r,'income'].mean()

print("\n----- Vérifications finales (après calibration ciblée) -----")
print(f"Moyenne globale finale: {final_global_mean:,.2f} DH (Cible: {global_target_mean:,.2f} DH)")
print(f"Moyenne urbaine finale: {final_urban_mean:,.2f} DH (Cible: {urban_target_mean:,.2f} DH)")
print(f"Moyenne rurale finale : {final_rural_mean:,.2f} DH (Cible: {rural_target_mean:,.2f} DH)")

final_pct_below_global = (df['income'] < final_global_mean).mean() * 100
final_pct_below_urban = (df.loc[mask_u, 'income'] < final_urban_mean).mean() * 100
final_pct_below_rural = (df.loc[mask_r, 'income'] < final_rural_mean).mean() * 100
print(f"\nRépartition des revenus : {final_pct_below_global:.1f}% < moyenne (urbain {final_pct_below_urban:.1f}%, rural {final_pct_below_rural:.1f}%)")
print("Cibles : 71,8% (urbain 65,9%, rural 85,4%)")
current_global_mean = df['income'].mean()  # Calculate the global mean
global_tgt = 21949  # Define the global target mean income
df['income'] = (df['income'] * (global_tgt/current_global_mean)).round().astype(int)

print("\nMoyennes par secteur d'emploi:")
print(df.groupby("secteur_emploi")["income"].mean().sort_values(ascending=False))


# --- BEGIN NEW BLOCK: Final Adjustment for Secondary Income Effect ---
print("\n----- Ajustement final pour l'effet du revenu secondaire -----")
mean_income_sec_yes_before = df[df['source_revenu_secondaire'] == 1]['income'].mean()
mean_income_sec_no_before = df[df['source_revenu_secondaire'] == 0]['income'].mean()

print(f"Moyenne AVANT ajustement (Revenu Secondaire Oui): {mean_income_sec_yes_before:,.0f} DH")
print(f"Moyenne AVANT ajustement (Revenu Secondaire Non): {mean_income_sec_no_before:,.0f} DH")

# Define how much higher the 'yes' group's mean income should be (e.g., 5% higher)
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
        df.loc[df['source_revenu_secondaire'] == 1, 'income'] = \
            (df.loc[df['source_revenu_secondaire'] == 1, 'income'] * adjustment_factor).round().astype(int)
        
        # Ensure incomes don't go below a certain minimum (e.g., 0 or a small positive value)
        df.loc[df['source_revenu_secondaire'] == 1, 'income'] = df.loc[df['source_revenu_secondaire'] == 1, 'income'].apply(lambda x: max(x, 100))


        mean_income_sec_yes_after = df[df['source_revenu_secondaire'] == 1]['income'].mean()
        mean_income_sec_no_after = df[df['source_revenu_secondaire'] == 0]['income'].mean() # Should be same as before this specific adjustment
        
        print(f"Moyenne APRÈS ajustement (Revenu Secondaire Oui): {mean_income_sec_yes_after:,.0f} DH")
        print(f"Moyenne APRÈS ajustement (Revenu Secondaire Non): {mean_income_sec_no_after:,.0f} DH")
        if mean_income_sec_no_after > 0:
             print(f"Ratio Oui/Non après ajustement: {(mean_income_sec_yes_after/mean_income_sec_no_after):.2f} (Cible: {desired_mean_ratio:.2f})")
    else:
        print("Ajustement non effectué (une des moyennes est zéro ou négative).")
else:
    print("Aucun ajustement nécessaire, la relation de revenu souhaitée pour le revenu secondaire est déjà respectée ou les données sont insuffisantes.")

# Note: This adjustment might slightly alter the overall global/urban/rural means.
# You can re-calculate and print them here if needed for final verification.
final_global_mean_after_all_adj = df['income'].mean()
print(f"\nMoyenne globale finale (après TOUS les ajustements): {final_global_mean_after_all_adj:,.2f} DH")
# --- END NEW BLOCK ---
# ==============================================
# 10. Sauvegarde du dataset
# ==============================================
df.to_csv("dataset_revenu_marocains.csv", index=False)
print("\nDataset généré avec succès !")