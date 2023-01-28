# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
# Module de fonctions utiles pour le cours
# Attention : il faut que le fichier soit placé dans le
# même répertoire que ce script
from clust_util import plotly_2d_clust, compute_inertia, plotly_2d_highlight_inertia


import os  # Fonctions systèmes utiles
import numpy as np  # Calcul numérique
import pandas as pd  # Manipulation des données
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation
import ipdb
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
example_desc = "Classification aléatoire"
var_index = "country"
var_x = "exports"
var_y = "imports"
var_cls = "cls"
var_text = "index"
cls_vlist = ["c1", "c2", "c3", "c4"]
cls_cmap = {cls: px.colors.qualitative.T10[i]
            for i, cls in enumerate(cls_vlist)}
between_inertia_color = "#7a1d90"

# Indique si l'on souhaite afficher les graphiques dans le navigateur web
show_plots = False


# Traitements
# -----------
# Filtrage des données
data_sel_df = data_df[[var_index, var_x, var_y]]\
    .set_index(var_index).sample(n=20, random_state=56860)
data_weights = 1/len(data_sel_df)

# On fixe la graîne du générateur de nombre aléatoire pour
# reproduire le même "hasard" d'une exécutation à l'autre
np.random.seed(56)

# Création d'une classification au hasard
data_sel_df[var_cls] = \
    np.random.choice(cls_vlist, len(data_sel_df))


# Calcul des inerties
# -------------------
IT, data_sel_center, tmp = \
    compute_inertia(data_sel_df[[var_x, var_y]],
                    weights=data_weights)

data_cls_grp = data_sel_df[[var_x, var_y]].groupby(data_sel_df[var_cls])

IW_cls_k = data_cls_grp.apply(
    lambda d: compute_inertia(d, weights=data_weights)[0])
IW_cls = IW_cls_k.sum()

cls_weights = data_cls_grp[var_x].count()/len(data_sel_df)
cls_centers = data_cls_grp.mean()

IB_cls, tmp, tmp = \
    compute_inertia(cls_centers,
                    weights=cls_weights)

PI_cls = 100*IB_cls/IT

cls_summary = pd.Series(
    [IT, IB_cls, IW_cls, PI_cls],
    index=["IT", "IB", "IW", "%I"],
    name="Clustering summary")

# Visualisations
# --------------
data_classif_fig = \
    plotly_2d_clust(data_df=data_sel_df,
                    data_weights=data_weights,
                    title=example_desc,
                    var_x=var_x,
                    var_y=var_y,
                    var_cls=var_cls,
                    var_text=var_text,
                    cls_vlist=cls_vlist,
                    cls_cmap=cls_cmap)

data_classif_iw_fig = \
    plotly_2d_clust(data_df=data_sel_df,
                    data_weights=data_weights,
                    title=example_desc + " (avec inertie intra)",
                    show_cls_center=True,
                    show_within_inertia_line=True,
                    show_between_inertia_line=False,
                    var_x=var_x,
                    var_y=var_y,
                    var_cls=var_cls,
                    var_text=var_text,
                    cls_vlist=cls_vlist,
                    cls_cmap=cls_cmap,
                    between_inertia_color=between_inertia_color)

data_classif_ib_fig = \
    plotly_2d_clust(data_df=data_sel_df,
                    data_weights=data_weights,
                    title=example_desc + " (avec inertie inter)",
                    show_cls_center=True,
                    show_within_inertia_line=False,
                    show_between_inertia_line=True,
                    var_x=var_x,
                    var_y=var_y,
                    var_cls=var_cls,
                    var_text=var_text,
                    cls_vlist=cls_vlist,
                    cls_cmap=cls_cmap,
                    between_inertia_color=between_inertia_color)

data_classif_inertia_fig = \
    plotly_2d_clust(data_df=data_sel_df,
                    data_weights=data_weights,
                    title=example_desc + " (avec inertie intra et inter)",
                    show_cls_center=True,
                    show_within_inertia_line=True,
                    show_between_inertia_line=True,
                    var_x=var_x,
                    var_y=var_y,
                    var_cls=var_cls,
                    var_text=var_text,
                    cls_vlist=cls_vlist,
                    cls_cmap=cls_cmap,
                    between_inertia_color=between_inertia_color)


if show_plots:
    pio.show(data_classif_fig)
