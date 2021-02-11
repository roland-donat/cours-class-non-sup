# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
# Module de fonctions utiles pour le cours
# Attention : il faut que le fichier soit placé dans le
# même répertoire que ce script
import ipdb
from clust_util import plotly_2d_clust, kmeans_algo, plotly_2d_clust_animation

import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation
import numpy as np  # Calcul numérique
from sklearn.cluster import KMeans  # Algorithme machine learnin
from sklearn.datasets import make_classification  # Génération de données

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
# var = [v for v in data_df.columns if v != var_index]
var = ["X", "Y"]
nb_var = 2
nb_data = 200
cluster_weights = [0.5, 0.2]
random_seed_data = 56

var_cls = "cls"
var_text = "index"
cls_val_template = "c{cls}"
nb_cls = 3
random_seed_init = 56
dist = "euclidean"
example_desc = "Moyennes mobiles - {nb_cls} classes - %IE = {PctI:.1%} - Dist {dist}"

case_study = {
    "euc": {"dist": "euclidean"},
    "mah": {"dist": "mahalanobis"},
}

# Indique si l'on souhaite afficher les graphiques dans le navigateur web
show_plots = True

# Traitements
# -----------

data_sel_df = \
    pd.DataFrame(
        make_classification(n_samples=nb_data,
                            n_features=nb_var,
                            n_informative=nb_var,
                            n_clusters_per_class=1,
                            n_classes=len(cluster_weights) + 1,
                            n_redundant=0,
                            weights=cluster_weights,
                            random_state=random_seed_data)[0],
        columns=var)

var_x = var[0]
var_y = var[1]


cls_vlist = [cls_val_template.format(cls=k)
             for k in range(nb_cls)]
cls_cmap = {cls: px.colors.qualitative.T10[i]
            for i, cls in enumerate(cls_vlist)}

data_classif_dfig = {}

for case, params in case_study.items():

    data_cls, cls_centers, cls_iw = \
        kmeans_algo(data_sel_df, nb_cls=len(cls_vlist),
                    iter_max=30,
                    var_cls=var_cls,
                    cls_labels=cls_vlist,
                    return_iter_info=True,
                    random_seed=random_seed_init,
                    **params)

    fig_title = example_desc.format(nb_cls=nb_cls,
                                    PctI=cls_iw["PctI"].iloc[-1],
                                    **params)

    data_classif_dfig[case] = \
        plotly_2d_clust_animation(
            data_df=data_sel_df,
            cls_values_it=data_cls,
            title=fig_title,
            show_cls_center=True,
            show_cls_ellipse=True,
            var_x=var_x,
            var_y=var_y,
            var_cls=var_cls,
            var_text=var_text,
            cls_vlist=cls_vlist,
            cls_cmap=cls_cmap,
            **params)

    if show_plots:
        pio.show(data_classif_dfig[case])
