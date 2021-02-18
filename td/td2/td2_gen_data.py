
import numpy as np
import pandas as pd
import sklearn.datasets as skd
import plotly.express as px

import ipdb

# Création de données circulaires
data_circ_arr, data_circ_cls_true = \
    skd.make_blobs(n_samples=1000, centers=4,
                   cluster_std=2, random_state=14)
data_circ_df = pd.DataFrame(data_circ_arr, columns=["X1", "X2"])

data_circ_fig = px.scatter(data_circ_df, x="X1", y="X2", width=750, height=750)

data_circ_df.to_csv("data_circ.csv", sep=";", index=False)


data_circ_fig.show()

# Transformation des données par application d'une matrice de rotation
# aléatoires
data_circ_df = pd.read_csv("data_circ.csv", sep=";")
var = data_circ_df.columns

rng = np.random.RandomState(13)


data_circ_grp = data_circ_df.groupby(data_circ_cls_true)

cls_resizing = [1, 0.5, 0.25, 0.75]

rotation_mat = pd.DataFrame(rng.randn(2, 2),
                            index=var,
                            columns=var)

# Transform cluster data
data_ellipse_df_list = []
for i, (cls, data_df) in enumerate(data_circ_grp):
    nb_cls_data = int(len(data_df)*cls_resizing[i])

    data_ellipse_df_list.append(
        data_df.sample(nb_cls_data, random_state=56) @ rotation_mat)

data_ellipse_df = pd.concat(data_ellipse_df_list,
                            axis=0)

data_ellipse_fig = px.scatter(
    data_ellipse_df, x="X1", y="X2", width=750, height=750)

data_ellipse_df.to_csv("data_ellipse.csv", sep=";", index=False)

data_ellipse_fig.show()
