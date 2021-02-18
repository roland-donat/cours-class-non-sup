# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
import plotly.express as px  # Visualisation
import plotly.io as pio  # Visualisation
from plotly.figure_factory import create_dendrogram  # Visualisation

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
example_desc = "Représentation d'une classification hiérarchique par dendrogramme"
cls_cmap = px.colors.qualitative.T10

# Indique si l'on souhaite afficher les graphiques dans le navigateur web
show_plots = False


# Traitements
# -----------
data_sel_df = data_df.set_index(var_index).sample(n=75, random_state=56860)

data_dendro_fig = create_dendrogram(data_sel_df,
                                    labels=data_sel_df.index,
                                    colorscale=cls_cmap,
                                    color_threshold=60000)

data_dendro_fig.update_layout(
    title_text=example_desc,
    width=1000, height=600)

if show_plots:
    pio.show(data_dendro_fig)
