# Modules nécessaire pour l'exécution du script
# ---------------------------------------------
import os  # Fonctions systèmes utiles
import pandas as pd  # Manipulation des données
import numpy as np
import scipy.spatial.distance as scd
import clust_util as clu

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
var = ["life_expec", "total_fer", "inflation"]
dist = "euclidean"

# Traitements
# -----------
data_sel_df = data_df.set_index(
    var_index)[var].sample(n=10, random_state=56860)

data_dist_mat = scd.cdist(data_sel_df, data_sel_df, metric=dist)
data_dist_mat_df = \
    pd.DataFrame(data_dist_mat,
                 index=data_sel_df.index,
                 columns=data_sel_df.index)


np.random.seed(56)
data_cls = np.random.choice(range(3), len(data_sel_df))
data_cls_s = pd.Series([f"C{cls+1}" for cls in data_cls],
                       index=data_sel_df.index,
                       name="cls")


data_sel_cls_grp = data_sel_df.groupby(data_cls_s)

data_cls_df1 = data_sel_cls_grp.get_group("C1")
data_cls_df2 = data_sel_cls_grp.get_group("C2")
data_cls_df3 = data_sel_cls_grp.get_group("C3")

dsl_12, dsl_12_idx, dsl_12_mat_df = \
    clu.dist_group_single(data_cls_df1, data_cls_df2,
                          dist=dist, returns_dist_mat=True)

dcl_13, dcl_13_idx, dcl_13_mat_df = \
    clu.dist_group_complete(data_cls_df1, data_cls_df3,
                            dist=dist, returns_dist_mat=True)

dga_23, dga_23_mat_df = \
    clu.dist_group_average(data_cls_df2, data_cls_df3,
                           dist=dist, returns_dist_mat=True)


data_mu_cls_df = data_sel_cls_grp.mean()

dward_12 = \
    clu.dist_group_ward(data_cls_df1, data_cls_df2)
dward_13 = \
    clu.dist_group_ward(data_cls_df1, data_cls_df3)
dward_23 = \
    clu.dist_group_ward(data_cls_df2, data_cls_df3)
