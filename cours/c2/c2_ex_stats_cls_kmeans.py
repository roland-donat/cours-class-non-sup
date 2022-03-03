# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
# Module de fonctions utiles pour le cours
# Attention : il faut que le fichier soit placé dans le
# même répertoire que ce script
from clust_util import plotly_2d_clust

import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation
from sklearn.cluster import KMeans

pio.renderers.default = 'browser'


# Chargement des données
# ----------------------
# Attention : dans cet exemple, les données sont cencées être dans
# le répertoire 'data' qui se trouve lui-même dans le répertoire
# de ce script
data_filename = os.path.join("data", "country_data.csv")
data_df = pd.read_csv(data_filename,
                      sep=",")

# Configuration de l'exemple
# --------------------------
var_index = "country"
var = ["exports", "health", "income"]
var_x = var[0]
var_y = var[1]
example_desc = f"Partition sur le nuage {var_x} vs {var_y} avec dispertion empirique"
var_cls = "cls"
var_text = "index"
cls_vlist = ["c1", "c2", "c3"]
cls_cmap = {cls: px.colors.qualitative.T10[i]
            for i, cls in enumerate(cls_vlist)}

# Indique si l'on souhaite afficher les graphiques dans le navigateur web
show_plots = False


# Traitements
# -----------
data_sel_df = data_df.set_index("country")[var]\
                     .sample(n=20, random_state=56860)

kmeans_model = KMeans(
    init="random",
    n_clusters=len(cls_vlist),
    n_init=1,
    max_iter=1,
    random_state=56
)

kmeans_model.fit(data_sel_df)
data_sel_df[var_cls] = kmeans_model.predict(data_sel_df)
# Change class labels
data_sel_df[var_cls].replace({i: cl for i, cl in enumerate(cls_vlist)},
                             inplace=True)


# Statistiques du partitionnement
data_cls_grp = data_sel_df.groupby(var_cls)

# Effectif
cls_count = \
    data_cls_grp["exports"].count()\
                           .rename("Effectif")

# Points moyens
cls_mean = data_cls_grp.mean()

# Matrice de variances
cls_cov = data_cls_grp.cov()

data_classif_fig = plotly_2d_clust(data_df=data_sel_df,
                                   title=example_desc,
                                   show_cls_center=True,
                                   show_cls_ellipse=True,
                                   var_x=var_x,
                                   var_y=var_y,
                                   var_cls=var_cls,
                                   var_text=var_text,
                                   cls_vlist=cls_vlist,
                                   cls_cmap=cls_cmap)


if show_plots:
    pio.show(data_classif_fig)
