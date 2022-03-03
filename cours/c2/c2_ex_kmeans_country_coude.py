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
import plotly.graph_objects as go  # Visualisation

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

# Variables : ['country', 'child_mort', 'exports',
#              'health', 'imports', 'income',
#              'inflation', 'life_expec', 'total_fer', 'gdpp']

# Configuration de l'exemple
# --------------------------
var_index = "country"
# var = [v for v in data_df.columns if v != var_index]
var = ["life_expec", "total_fer"]
var_x = var[0]
var_y = var[1]
var_cls = "cls"
var_text = "index"
cls_val_template = "c{cls}"
nb_cls_list = range(2, 10)
nb_data = None
random_seed_init = 56
random_seed_data = 56
dist = "euclidean"
example_desc = "Moyennes mobiles - Évolution du % d'inertie expliquée par la partition en fonction du nombre de classes"
COLORS = {
    "primary": "#7a1d90",
    "secondary": "#9e51ae",
    "tertiary": "#b370be",
    "quaternary": " #ff8e71",
    "quinary": " #9f5f80",
    "alert": " #a10010",
}


# Indique si l'on souhaite afficher les graphiques dans le navigateur web
show_plots = False

# Traitements
# -----------
data_sel_df = data_df.set_index(var_index)[var]
if not(nb_data is None):
    data_sel_df = data_sel_df.sample(n=nb_data,
                                     random_state=random_seed_data)


IW_k_s = pd.Series(0, index=nb_cls_list, name="%IE")
IW_k_s.index.name = "# classes"
for nb_cls in nb_cls_list:

    cls_vlist = [cls_val_template.format(cls=k)
                 for k in range(nb_cls)]

    data_cls, cls_centers, cls_iw = \
        kmeans_algo(data_sel_df, nb_cls=len(cls_vlist),
                    iter_max=30,
                    var_cls=var_cls,
                    cls_labels=cls_vlist,
                    dist=dist,
                    return_iter_info=False,
                    random_seed=random_seed_init
                    )

    # # Try to reproduce results with Sklearn's kmeans
    # kmeans_skl = KMeans(n_clusters=nb_cls,
    #                     n_init=1,
    #                     init='random',
    #                     random_state=random_seed_init)
    # cluster_labels = \
    #     kmeans_skl.fit_predict(data_sel_df)

    IW_k_s.loc[nb_cls] = cls_iw["PctI"]


IW_k_fig = go.Figure()

# Add traces
hovertemplate = \
    f'{IW_k_s.index.name} = %{{x}}<br>'\
    f'{IW_k_s.name} = %{{y:.1%}}'

IW_k_fig.add_scatter(x=IW_k_s.index,
                     y=IW_k_s,
                     mode='lines+markers',
                     marker_color=COLORS["primary"],
                     marker_size=10,
                     line_color=COLORS["secondary"],
                     hovertemplate=hovertemplate)

IW_k_fig.update_layout(title=example_desc,
                       xaxis_title=IW_k_s.index.name,
                       yaxis_title=IW_k_s.name)

if show_plots:

    pio.show(IW_k_fig)
