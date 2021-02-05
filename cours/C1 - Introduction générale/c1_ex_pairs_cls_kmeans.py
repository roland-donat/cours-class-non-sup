# Modules nécessaire pour l'exécution du script
# ---------------------------------------------

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
example_desc = "Nuage de points multivarié (pairs plot) avec classification"
var_index = "country"
var = ["exports", "health", "income"]
var_cls = "cls"
var_text = "index"
cls_vlist = ["c1", "c2", "c3"]
cls_cmap = {cls: px.colors.qualitative.T10[i]
            for i, cls in enumerate(cls_vlist)}

# Indique si l'on souhaite afficher les graphiques dans le navigateur web
show_plots = False


# Traitements
# -----------
data_sel_df = data_df.set_index("country")

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

data_classif_fig = \
    px.scatter_matrix(data_sel_df,
                      title=example_desc,
                      dimensions=var,
                      color=var_cls,
                      category_orders={var_cls: cls_vlist},
                      color_discrete_map=cls_cmap)

data_classif_fig.update_traces(
    diagonal_visible=False,
)

data_classif_fig.update_xaxes(
    showline=True
)


if show_plots:
    pio.show(data_classif_fig)
