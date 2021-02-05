import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial.distance import pdist, cdist, squareform

data_path = "https://roland-donat.github.io/cours-class-non-sup/td/td1/wine.csv"
data_df = pd.read_csv(data_path, sep=",")

# Affichage d'informations générales
print(data_df.describe())

print(data_df.info())

px.box(data_df, 
       title="Boxplot de chaque variable des données d'analyse des vins").show()

# Centrage et réduction des données
data_scaled_df = (data_df - data_df.mean()).div(data_df.std())

px.box(data_scaled_df, 
       title="Boxplots des données d'analyse des vins centrées-reduites").show()

# Calcul de la matrice de corrélation
data_scaled_corr_mat = data_scaled_df.corr()
px.imshow(data_scaled_corr_mat, 
          title="Matrice de corrélations linéaires des données d'analyse des vins").show()

px.scatter_matrix(data_scaled_df, 
                  title="Diagramme en paires des données d'analyse des vins").show()

# Distances euclidiennes entre les trois premiers individus
d01 = ((data_scaled_df.iloc[0] - data_scaled_df.iloc[1])**2).sum()**(0.5)
d02 = ((data_scaled_df.iloc[0] - data_scaled_df.iloc[2])**2).sum()**(0.5)
d12 = ((data_scaled_df.iloc[1] - data_scaled_df.iloc[2])**2).sum()**(0.5)

# Vérification de l'inégalité triangulaire
print("Vérification de l'inégalité triangulaire !") \
    if (d01 + d12) >= d12 \
    else print("Les mathématiques s'effondrent...")

# Calcul des distances des observations deux à deux sous forme unidimensionnelle
data_dvec_euc = pdist(data_scaled_df, metric="euclidean")
# Mise en forme de matrice carrée
data_dmat_euc = squareform(data_dvec_euc)
# Conversion en DataFrame pour le confort
data_dmat_euc_df = pd.DataFrame(data_dmat_euc,
                                index=data_scaled_df.index,
                                columns=data_scaled_df.index)

data_dmat_euc_df.iloc[0,1] == d01
data_dmat_euc_df.iloc[0,2] == d02
data_dmat_euc_df.iloc[1,2] == d12

mu_data = data_scaled_df.mean()
d2_data_mu = ((data_scaled_df - mu_data)**2).sum(axis=1)
I_T = d2_data_mu.sum()

data_var_sum = data_scaled_df.var().sum()

# Le rapport entre l'inertie totale et la somme des variances des variables donne
# la taille du jeu de donnée - 1 car :
# - nous sommes dans le cas équipondéré avec poids = 1
# - la méthode var() calcule la variance empirique corrigée 
#   Rappel : variance empirique corrigée = effectif/(effectif-1) * variance empirique
print(I_T/data_var_sum)

classif_name = "CA"
print("\nClassification {classif_name}\n")
print("-----------------\n")


# Création d'une nouvelle variable initialisée à la valeur c1
data_scaled_df["cls_CA"] = "c1"
# Affectation de la classe 1 aux individus 50-99
data_scaled_df["cls_CA"].iloc[50:100] = "c2"
# Affectation de la classe 2 aux individus restant
data_scaled_df["cls_CA"].iloc[100:] = "c3"

# Visualisation
px.scatter(data_scaled_df,
           x="OD280", y="Alcohol",
           color="cls_CA",
           title=f"OD280 vs Alcohol avec classification {classif_name}").show()

# Regroupement des données par classe
data_cls_CA_grp = data_scaled_df.groupby("cls_CA")

# Calcul des moyennes (centres de gravité de chaque classe)
mu_cls_CA = data_cls_CA_grp.mean()

I_cls_CA = pd.Series(0, index=mu_cls_CA.index, name="I_cls_CA")
for cls_CA, data_cls_CA_df in data_cls_CA_grp:
    # Calcul des distances au carré entre chaque individu de la classe et son centre
    d2_data_cls_CA = ((data_cls_CA_df - mu_cls_CA.loc[cls_CA])**2).sum(axis=1)
    # Déduction de l'inertie interne à la classe
    I_cls_CA.loc[cls_CA] = d2_data_cls_CA.sum()

# Inertie intra-classe de la classification
I_W_CA = I_cls_CA.sum()

# Déduction de l'inertie inter-classe
print(f"I_B({classif_name}) = {I_T - I_W_CA}")

# Calcul du % d'inertie expliquée par la classification
PI_CA = 100*(1 - I_W_CA/I_T)

# Calcul de la somme des variances des variables au sein de chaque classe
data_cls_CA_var_sum = data_cls_CA_grp.var().sum(axis=1)

# Le rapport entre l'inertie interne et la variance interne donne
# l'effectif de la classe - 1 car
# - nous sommes dans le cas équipondéré avec poids = 1
# - la méthode var() calcule la variance empirique corrigée 
#   Rappel : variance empirique corrigée = effectif/(effectif-1) * variance empirique
print(I_cls_CA/data_cls_CA_var_sum)

# Effectif de chaque classe - 1
N_cls_CA = data_cls_CA_grp["Alcohol"].count()
# Somme de la somme des variances empiriques corrigées des variables de chaque classe
# pondérée par l'effectif de la classe
I_W_CA_bis = ((N_cls_CA - 1)*data_cls_CA_var_sum).sum()

# Calcul des distances au carré entre les centres de gravité de classe et
# le centre de gravité des données
e2_mu_cls_CA_fact = (mu_cls_CA - mu_data)**2
d2_mu_cls_CA_data = e2_mu_cls_CA_fact.sum(axis=1)
I_B_CA = (N_cls_CA*d2_mu_cls_CA_data).sum()

# Complément lien avec la variance
# Calcul de la variance des variables dans le cas où l'on remplace chaque
# individu par le centre de sa classe
d2_mu_cls_CA_fact = (mu_cls_CA - mu_data)**2
mu_cls_CA_var = (N_cls_CA @ e2_mu_cls_CA_fact)/(len(data_scaled_df) - 1)
mu_cls_CA_var_sum = mu_cls_CA_var.sum()

# == effectif - 1
print(f"Rapport entre I_B({classif_name}) et somme des variances inter = {I_B_CA/mu_cls_CA_var_sum}")

classif_name = "CB"
print("\nClassification {classif_name}\n")
print("-----------------\n")

# On fixe la graîne du générateur de nombre aléatoire pour 
# reproduire le même "hasard" d'une exécutation à l'autre
np.random.seed(56)

# Initalisation au hasard de la classification
data_scaled_df["cls_CB"] = np.random.choice(["c1", "c2", "c3"],
                                            len(data_scaled_df))

# Visualisation
px.scatter(data_scaled_df,
           x="OD280", y="Alcohol",
           color="cls_CB",
           title=f"OD280 vs Alcohol avec classification {classif_name}").show()

# Regroupement des données par classe
data_cls_CB_grp = data_scaled_df.groupby("cls_CB")

# Calcul des moyennes (centres de gravité de chaque classe)
mu_cls_CB = data_cls_CB_grp.mean()

I_cls_CB = pd.Series(0, index=mu_cls_CB.index, name="I_cls_CB")
for cls_CB, data_cls_CB_df in data_cls_CB_grp:
    # Calcul des distances au carré entre chaque individu de la classe et son centre
    d2_data_cls_CB = ((data_cls_CB_df - mu_cls_CB.loc[cls_CB])**2).sum(axis=1)
    # Déduction de l'inertie interne à la classe
    I_cls_CB.loc[cls_CB] = d2_data_cls_CB.sum()

# Inertie intra-classe de la classification
I_W_CB = I_cls_CB.sum()

# Déduction de l'inertie inter-classe
print(f"I_B({classif_name}) = {I_T - I_W_CB}")

# Calcul du % d'inertie expliquée par la classification
PI_CB = 100*(1 - I_W_CB/I_T)

# Calcul de la somme des variances des variables au sein de chaque classe
data_cls_CB_var_sum = data_cls_CB_grp.var().sum(axis=1)

# Le rapport entre l'inertie interne et la variance interne donne
# l'effectif de la classe - 1 car
# - nous sommes dans le cas équipondéré avec poids = 1
# - la méthode var() calcule la variance empirique corrigée 
#   Rappel : variance empirique corrigée = effectif/(effectif-1) * variance empirique
print(I_cls_CB/data_cls_CB_var_sum)

# Effectif de chaque classe - 1
N_cls_CB = data_cls_CB_grp["Alcohol"].count()
# Somme de la somme des variances empiriques corrigées des variables de chaque classe
# pondérée par l'effectif de la classe
I_W_CB_bis = ((N_cls_CB - 1)*data_cls_CB_var_sum).sum()

# Calcul des distances au carré entre les centres de gravité de classe et
# le centre de gravité des données
e2_mu_cls_CB_fact = (mu_cls_CB - mu_data)**2
d2_mu_cls_CB_data = e2_mu_cls_CB_fact.sum(axis=1)
I_B_CB = (N_cls_CB*d2_mu_cls_CB_data).sum()

# Complément lien avec la variance
# Calcul de la variance des variables dans le cas où l'on remplace chaque
# individu par le centre de sa classe
d2_mu_cls_CB_fact = (mu_cls_CB - mu_data)**2
mu_cls_CB_var = (N_cls_CB @ e2_mu_cls_CB_fact)/(len(data_scaled_df) - 1)
mu_cls_CB_var_sum = mu_cls_CB_var.sum()

# == effectif - 1
print(f"Rapport entre I_B({classif_name}) et somme des variances inter = {I_B_CB/mu_cls_CB_var_sum}")
