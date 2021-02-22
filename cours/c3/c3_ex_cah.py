# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
import numpy as np
import clust_util as clu
import sklearn.cluster as skc
import plotly.graph_objects as go  # Visualisation
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation
from plotly.figure_factory import create_dendrogram  # Visualisation

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
var = ["life_expec", "total_fer"]
dist = "euclidean"
index_sel = ["France", "Mozambique",
             "Guatemala", "Brazil", "Japan", "Argentina", "Senegal"]

COLORS = {
    "primary": "#7a1d90",
    "secondary": "#9e51ae",
    "tertiary": "#b370be",
    "quaternary": " #ff8e71",
    "quinary": " #9f5f80",
    "alert": " #a10010",
}

show_plots = False
save_fig = False

# Traitements
# -----------
data_sel_df = data_df.set_index(
    var_index)[var].loc[index_sel]  # .sample(n=5, random_state=56)


# Déroulement CAH
cls_list, cls_dist_mat_list, aggreg_cls_list, aggreg_dist_list, inertia_within, inertia_ratio = \
    clu.cah(data_sel_df)

# CAH avec sklearn
# Attention : les distances de Ward calculé correspond à sqrt(2*D^2) avec
# D les distances de ward calculé par notre algorithme
# cah_model = skc.AgglomerativeClustering(distance_threshold=0,
#                                         n_clusters=None,
#                                         affinity='euclidean',
#                                         linkage='ward')
# cah_model.fit(data_sel_df)

inertia_within_k_fig = go.Figure()

# Add traces
hovertemplate = \
    f'# classes = %{{x}}<br>'\
    f'IW = %{{y:.1f}}'

inertia_within_k_fig.add_scatter(
    x=list(range(len(data_sel_df), 1, -1)),
    y=inertia_within,
    mode='lines+markers',
    marker_color=COLORS["primary"],
    marker_size=10,
    line_color=COLORS["secondary"],
    hovertemplate=hovertemplate)

inertia_within_k_fig.update_layout(
    title_font_size=20,
    font_size=14,
    title="Inertie intra-classe vs # classes",
    xaxis_title="# classes",
    yaxis_title="IW")

inertia_ratio_k_fig = go.Figure()

# Add traces
hovertemplate = \
    f'# classes = %{{x}}<br>'\
    f'%IE = %{{y:.1%}}'

inertia_ratio_k_fig.add_scatter(
    x=list(range(len(data_sel_df), 1, -1)),
    y=inertia_ratio,
    mode='lines+markers',
    marker_color=COLORS["primary"],
    marker_size=10,
    line_color=COLORS["secondary"],
    hovertemplate=hovertemplate)

inertia_ratio_k_fig.update_layout(
    title_font_size=20,
    font_size=14,
    title="% Inertie expliquée vs # classes",
    xaxis_title="# classes",
    yaxis_title="IW")


# Visualisations
if save_fig:
    data_fig = px.scatter(data_sel_df,
                          x=var[0],
                          y=var[1],
                          text=data_sel_df.index,
                          width=1000, height=1000)

    data_fig.update_traces(
        textposition='top center',
        marker=dict(size=16,
                    color=COLORS["quinary"],
                    line=dict(width=1,
                              color=COLORS["primary"])))

    data_fig.update_layout(
        title_font_size=40,
        font_size=20,
        title_text=f'Données natalité vs espérance de vie'
    )

    data_fig.write_image("fig/cah_data_ori.svg")

    data_dendro_fig = create_dendrogram(data_sel_df,
                                        labels=data_sel_df.index)

    data_dendro_fig.update_layout(
        title_text="Dendrogramme de la CAH",
        title_font_size=20,
        font_size=16,
        width=1000, height=1000
    )

    data_dendro_fig.write_image("fig/cah_dendro_final.svg")

if show_plots:

    pio.show(inertia_within_k_fig)
    pio.show(inertia_ratio_k_fig)
