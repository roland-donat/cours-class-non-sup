# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
# Module de fonctions utiles pour le cours
# Attention : il faut que le fichier soit placé dans le
# même répertoire que ce script
from clust_util import plotly_2d_highlight_inertia, compute_inertia

import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
import numpy as np   # Calcul numérique
# from scipy.spatial.distance import pdist, squareform  # Calcul numérique
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation
import plotly.graph_objects as go  # Visualisation

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
var = ["exports", "imports"]
data_weights = 1/len(data_df)
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
# Selection des données
data_sel_df = data_df.set_index(var_index)[var]

var_x = var[0]
var_y = var[1]

# Extract observation from data
data_extract_df_list = []
data_extract_df_list.append(data_sel_df.sample(n=5, random_state=56860))
data_extract_df_list.append(data_sel_df.sample(n=10, random_state=56860))


# Génération des points de référence pour le calcul de l'inertie
np.random.seed(56860)
point_ref_list = \
    [pd.Series(np.random.uniform(0, 100, 2),
               index=var) for i in range(3)]


inertia_res = []

for data_extract_df in data_extract_df_list:
    inertia_res_bis = []
    for point_ref in point_ref_list:

        # Calcul inertie
        inertia_val, tmp, data_inertia = \
            compute_inertia(data_extract_df,
                            point_ref=point_ref,
                            weights=data_weights)

        # Visualisations
        inertia_fig = px.scatter(data_sel_df,
                                 x=data_sel_df[var_x],
                                 y=data_sel_df[var_y])

        inertia_fig.update_traces(
            marker=dict(size=8,
                        color=COLORS["quinary"],
                        opacity=0.25))

        inertia_fig = \
            plotly_2d_highlight_inertia(
                inertia_fig,
                data_extract_df,
                point_ref=point_ref,
                annote_inertia=False,
                inertia_params=dict(
                    weights=data_weights
                ),
                inertia_text_props=dict(),
                data_marker_props=dict(
                    marker_color=COLORS["quinary"]),
                line_props=dict(
                    name="Inertia lines",
                    line_color=COLORS["primary"]
                ))

        inertia_fig.update_layout(
            title_text='Inertia lines from point '
            f'({point_ref[var_x]:.2f}, {point_ref[var_y]:.2f})')

        inertia_res_bis.append({"value": inertia_val,
                                "contrib": data_inertia,
                                "fig": inertia_fig})

    inertia_res.append(inertia_res_bis)


# Inertie totale
data_it_fig = px.scatter(data_sel_df,
                         x=data_sel_df[var_x],
                         y=data_sel_df[var_y])

data_it_fig.update_traces(
    marker=dict(size=8,
                color=COLORS["quinary"],
                opacity=0.25))

data_it_fig = \
    plotly_2d_highlight_inertia(
        data_it_fig,
        data_sel_df,
        annote_inertia=False,
        inertia_params=dict(
            weights=data_weights
        ),
        inertia_text_props=dict(),
        data_marker_props=dict(
            marker_color=COLORS["quinary"]),
        line_props=dict(
            name="Inertia lines",
            line_color=COLORS["primary"],
            legendgroup="lines"
        ))

data_it_fig.update_layout(
    title_text='Total inertia lines')

if show_plots:

    fig = go.Figure().set_subplots(len(data_extract_df_list),
                                   len(point_ref_list),
                                   horizontal_spacing=0.1)

    for i in range(len(data_extract_df_list)):
        for j in range(len(point_ref_list)):
            [fig.add_trace(fig_data, row=i+1, col=j+1)
             for fig_data in inertia_res[i][j]["fig"]["data"]]

    pio.show(data_it_fig)
