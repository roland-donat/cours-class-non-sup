import os
import math
import numpy as np
import pandas as pd
import tabulate
import matplotlib
import plotly.express as px
import plotly.io as pio
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.cluster import KMeans

COLORS = {
    "primary": "#7a1d90",
    "secondary": "#9e51ae",
    "tertiary": "#b370be",
    "quaternary": " #ff8e71",
    "quinary": " #9f5f80",
    "alert": " #a10010",
}

data_filename = os.path.join("data", "country_data.csv")
data_df = pd.read_csv(data_filename,
                        sep=",")

data_3v_head_df = data_df.head(6)[["country", "exports", "health", "income"]]

data_3v_head_cls_df = data_3v_head_df.copy()

data_3v_head_cls_df["class"] = ["C3", "C3", "C1", "C2", "C2", "C3"]

data_styles = {"class": [dict(selector="td", props=[("color", "#7a1d90")])]}

data_3v_head_cls_df.style.set_table_styles(data_styles).render()

data_3v_cls_df = data_df[["country", "exports", "health", "income"]].set_index("country")

kmeans_model = KMeans(
    init="random",
    n_clusters=3,
    n_init=1,
    max_iter=1,
    random_state=56
)

kmeans_model.fit(data_3v_cls_df)
classes = kmeans_model.predict(data_3v_cls_df)

data_3v_cls_df["class"] = [f"C{c+1}" for c in classes]

data_3v_cls_fig = \
    px.scatter_matrix(data_3v_cls_df, 
                      title="Nuage de points multivarié (pairs plot) avec classification",
                      dimensions=["exports", "health", "income"],
                      color="class")

data_3v_cls_fig.update_traces(
    diagonal_visible=False,
)

data_3v_cls_fig.update_xaxes(
    showline=True
)


pio.to_html(data_3v_cls_fig, include_plotlyjs="cdn",
            full_html=False,
            config={'displayModeBar': False})

data_3v_head_cls_df["class"] = ["C3", "C3", "C1", "C2", "?", "?"]

props = [('font-size', '16px')]
data_styles = [dict(selector="th", props=props),
               dict(selector="td", props=props)]
data_3v_head_cls_df.style\
                   .hide_index()\
                   .set_table_styles(data_styles).render()

data_3v_head_cls_df["class"] = ["?", "?", "?", "?", "?", "?"]

props = [('font-size', '16px')]
data_styles = [dict(selector="th", props=props),
               dict(selector="td", props=props)]
data_3v_head_cls_df.style\
            .hide_index()\
            .set_table_styles(data_styles).render()

data_3v_20_df = data_3v_cls_df.head(20)

# Statistiques sur les classes
# On regroupe les données sur la variable "class"
data_3v_20_class_grp =  data_3v_20_df.groupby("class")
nb_classes = len(data_3v_20_class_grp)
class_names = list(data_3v_20_class_grp.groups)                  

# Effectif
data_3v_20_class_count =  data_3v_20_class_grp["exports"].count()\
                                                         .rename("Effectif")\
                                                         .to_frame()

# Points moyens
data_3v_20_class_mean =  data_3v_20_class_grp.mean()

# Matrice de variances
data_3v_20_class_cov =  data_3v_20_class_grp.cov()

var_x = "exports"
var_y = "health"
var_class = "class"

data_3v_20_class_cmap = {class_names[i]: px.colors.qualitative.T10[i] 
                         for i in range(nb_classes)}
data_3v_20_fig = px.scatter(data_3v_20_df, 
                            x=var_x,
                            y=var_y,
                            text=data_3v_20_df.index,
                            color=var_class,
                            category_orders={var_class: list(data_3v_20_class_cmap.keys())},
                            color_discrete_map=data_3v_20_class_cmap)

data_3v_20_fig.update_traces(
    textposition='top center',
    marker=dict(size=10))

data_3v_20_fig.update_layout(
    hovermode=False,
    title_text=f'{var_x} vs {var_y}'
)


def compute_ellipse(variance_x=1, variance_y=1, cov_xy=0, 
                    mean_x=0, mean_y=0,
                    conf_alpha=0.95, nb_pts=100):
    term1 = (variance_x + variance_y)/2
    term2 = (((variance_x - variance_y)/2)**2 + cov_xy**2)**(0.5)
    
    l1 = term1 + term2
    l2 = term1 - term2
    
    theta = 0 if (cov_xy == 0) and (variance_x >= variance_y) \
        else np.pi/2 if (cov_xy == 0) and (variance_x < variance_y) \
        else np.arctan2(l1 - variance_x, cov_xy)
    
    conf = chi2.ppf(conf_alpha, 2)
    rot_x1 = (conf*l1)**(0.5)*np.cos(theta)
    rot_x2 = (conf*l2)**(0.5)*np.sin(theta)
    rot_y1 = (conf*l1)**(0.5)*np.sin(theta)
    rot_y2 = (conf*l2)**(0.5)*np.cos(theta)
    
    range_t = np.linspace(0, 2*np.pi, nb_pts)
    ell_x = [rot_x1*np.cos(t) - rot_x2*np.sin(t) + mean_x for t in range_t]
    ell_y = [rot_y1*np.cos(t) - rot_y2*np.sin(t) + mean_y for t in range_t]
    
    return ell_x, ell_y


def compute_class_ellipses(data_df, var_x, var_y, var_class, 
                           conf_alpha=0.95, nb_pts=100):
    data_class_grp =  data_df[[var_x, var_y, var_class]].groupby(var_class) 
    nb_classes = len(data_class_grp)                              
    class_names = list(data_class_grp.groups)                  
    # Class centers                                               
    data_class_mean =  data_class_grp.mean()                      
    
    # Class covariance matrix                                     
    data_class_cov =  data_class_grp.cov()                        
    
    class_ellipses = {}
    for class_cur in class_names:
                                                                  
        cov_xy = data_class_cov.loc[(class_cur, var_x), var_y]                               
        variance_x = data_class_cov.loc[(class_cur, var_x), var_x]                               
        variance_y = data_class_cov.loc[(class_cur, var_y), var_y]
        
        mean_x = data_class_mean.loc[class_cur, var_x]
        mean_y = data_class_mean.loc[class_cur, var_y]
        
        ell_x, ell_y = compute_ellipse(variance_x, variance_y, cov_xy, 
                                       mean_x, mean_y,
                                       conf_alpha, nb_pts)
        
        class_ellipses[class_cur] = {"x": ell_x, "y": ell_y}
    
    return class_ellipses


def plotly_add_class_ellipses(fig, data_df, var_x, var_y, 
                              var_class, conf_alpha=0.95, nb_pts=100,
                              class_color_map=None,
                              ellipse_opacity=0.75):
    
    class_ellipses = \
        compute_class_ellipses(data_df, var_x, var_y, var_class, 
                               conf_alpha, nb_pts)
    import ipdb
    ipdb.set_trace()
    for class_cur, class_ellipse in class_ellipses.items():
        fig.add_scatter(mode="lines",
                        **class_ellipse)
        
    return fig

data_3v_20_fig = \
    plotly_add_class_ellipses(
        fig=data_3v_20_fig,
        data_df=data_3v_20_df,
        var_x=var_x,
        var_y=var_y,
        var_class=var_class,
        conf_alpha=0.95,
        nb_pts=100,
        class_color_map=data_3v_20_class_cmap,
        ellipse_opacity=0.5)

pio.to_html(data_3v_20_fig, include_plotlyjs="cdn",
            full_html=False,
            config={'displayModeBar': False})

data_2d_sample_df = data_df[["country", "exports", "imports"]]\
    .set_index("country")\
    .head(10)

data_2d_sample_euclidean_dmat = pd.DataFrame(
    squareform(pdist(data_2d_sample_df, metric="euclidean")),
    index=data_2d_sample_df.index,
    columns=data_2d_sample_df.index)

data_3d_sample_df = data_df[["country", "exports", "imports", "income"]]\
    .set_index("country")\
    .head(10)

data_3d_sample_euclidean_dmat = pd.DataFrame(
    squareform(pdist(data_3d_sample_df, metric="euclidean")),
    index=data_3d_sample_df.index,
    columns=data_3d_sample_df.index)

fig = px.scatter(data_2d_sample_df, 
                 x="exports", 
                 y="imports",
                 text=data_2d_sample_df.index)

fig.update_traces(
    textposition='top center',
    marker=dict(size=12,
                color=COLORS["quinary"],
                line=dict(width=1,
                          color=COLORS["primary"])))

fig.update_layout(
    hovermode=False,
    title_text='Exportations (%PIB) vs Importations (%PIB)'
)

pio.to_html(fig, include_plotlyjs="cdn",
            full_html=False,
            config={'displayModeBar': False})

data_3d_sample_scatter = \
    px.scatter_3d(data_3d_sample_df, 
                  x="exports", 
                  y="imports",
                  z="income",
                  text=data_3d_sample_df.index)

data_3d_sample_scatter.update_traces(
    textposition='top center',
    marker=dict(size=8,
                opacity=0.8))

data_3d_sample_scatter.update_layout(
    hovermode=False,
    title_text='Exportations (%PIB) vs Importations (%PIB)'
)
