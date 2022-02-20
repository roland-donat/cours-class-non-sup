# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
from scipy.spatial.distance import pdist, squareform  # Calcul numérique
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation

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
var_2d = ["exports", "imports"]
var_3d = ["exports", "imports", "income"]
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
data_2d_df = data_df.set_index(var_index)[var_2d]\
                    .head(10)
data_3d_df = data_df.set_index(var_index)[var_3d]\
                    .head(10)

# Calcul des distances
data_2d_euc_dmat = pd.DataFrame(
    squareform(pdist(data_2d_df, metric="euclidean")),
    index=data_2d_df.index,
    columns=data_2d_df.index)


data_3d_euc_dmat = pd.DataFrame(
    squareform(pdist(data_3d_df, metric="euclidean")),
    index=data_3d_df.index,
    columns=data_3d_df.index)

data_3d_mah_dmat = pd.DataFrame(
    squareform(pdist(data_3d_df, metric="mahalanobis")),
    index=data_3d_df.index,
    columns=data_3d_df.index)


# Visualisations
data_2d_fig = px.scatter(data_2d_df,
                         x=var_2d[0],
                         y=var_2d[1],
                         text=data_2d_df.index)

data_2d_fig.update_traces(
    textposition='top center',
    marker=dict(size=12,
                color=COLORS["quinary"],
                line=dict(width=1,
                          color=COLORS["primary"])))

data_2d_fig.update_layout(
    title_text=f'{var_2d[0]} vs {var_2d[1]}'
)


if show_plots:
    pio.show(data_2d_fig)
