# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
# Module de fonctions utiles pour le cours
# Attention : il faut que le fichier soit placé dans le
# même répertoire que ce script
from clust_util import plotly_2d_clust, kmeans_algo, plotly_2d_clust_animation

import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation
import numpy as np  # Calcul numérique
from sklearn.cluster import KMeans  # Algorithme machine learning

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
var = ["imports", "exports"]
var_x = var[0]
var_y = var[1]
example_desc = "Moyennes mobiles - {nb_cls} centres - %IE = {PctI:.1%} (convergence)"
dist = "euclidean"
var_cls = "cls"
# var_text = "index"
var_text = None
cls_val_template = "c{cls}"
nb_cls_max = 6
random_seed = 56860

# Indique si l'on souhaite afficher les graphiques dans le navigateur web
show_plots = False


# Traitements
# -----------
data_sel_df = data_df.set_index("country")[var]


data_classif_dfig = {}
for nb_cls_cur in range(2, nb_cls_max + 1):

    cls_vlist = [cls_val_template.format(cls=k)
                 for k in range(nb_cls_cur)]
    cls_cmap = {cls: px.colors.qualitative.T10[i]
                for i, cls in enumerate(cls_vlist)}

    data_cls, cls_centers, cls_iw = \
        kmeans_algo(data_sel_df, nb_cls=len(cls_vlist),
                    dist=dist,
                    iter_max=30,
                    var_cls=var_cls,
                    cls_labels=cls_vlist,
                    return_iter_info=True,
                    random_seed=random_seed)

    data_classif_dfig[nb_cls_cur] = \
        plotly_2d_clust_animation(
            data_df=data_sel_df,
            cls_values_it=data_cls,
            title=example_desc.format(nb_cls=nb_cls_cur,
                                      PctI=cls_iw["PctI"].iloc[-1]),
            show_cls_center=True,
            show_cls_ellipse=True,
            var_x=var_x,
            var_y=var_y,
            var_cls=var_cls,
            var_text=var_text,
            cls_vlist=cls_vlist,
            cls_cmap=cls_cmap)


# ipdb.set_trace()

# kmeans_model = KMeans(
#     init="random",
#     n_clusters=len(cls_vlist),
#     n_init=1,
#     max_iter=1,
#     random_state=56
# )

# kmeans_model.fit(data_sel_df)
# data_sel_df[var_cls] = kmeans_model.predict(data_sel_df)
# # Change class labels
# data_sel_df[var_cls].replace({i: cl for i, cl in enumerate(cls_vlist)},
#                              inplace=True)


# # Statistiques du partitionnement
# data_cls_grp = data_sel_df.groupby(var_cls)

# # Effectif
# cls_count = \
#     data_cls_grp["exports"].count()\
#                            .rename("Effectif")

# # Points moyens
# cls_mean = data_cls_grp.mean()

# # Matrice de variances
# cls_cov = data_cls_grp.cov()

# data_classif_fig = plotly_2d_clust(data_df=data_sel_df,
#                                    title=example_desc,
#                                    show_cls_center=True,
#                                    show_cls_ellipse=True,
#                                    var_x=var_x,
#                                    var_y=var_y,
#                                    var_cls=var_cls,
#                                    var_text=var_text,
#                                    cls_vlist=cls_vlist,
#                                    cls_cmap=cls_cmap)


if show_plots:
    pio.show(data_classif_dfig[3])
