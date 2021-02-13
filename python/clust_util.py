import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import cdist
import plotly.graph_objects as go  # Visualisation


def hex_to_rgb(hexcode):
    hexcode = hexcode.lstrip("#")
    return tuple(int(hexcode[i:i+2], 16) for i in (0, 2, 4))


def compute_weighted_mean(data_df,
                          weights=None):

    if weights is None:
        weights = pd.Series(1, index=data_df.index)
    if isinstance(weights, (int, float)):
        weights = pd.Series(weights, index=data_df.index)

    weights_sum = weights.sum()
    data_weighted = data_df.multiply(weights, axis=0)

    return data_weighted.sum()/weights_sum, weights_sum


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


def compute_2d_clust_ellipses(data_df,
                              var_cls,
                              var_x=None,
                              var_y=None,
                              conf_alpha=0.95, nb_pts=100):

    if var_x is None:
        var_x = data_df.columns[0]
    if var_y is None:
        var_y = data_df.columns[1]

    data_cls = data_df[var_cls]
    data_class_grp = data_df[[var_x, var_y]].groupby(data_cls)

    class_names = list(data_class_grp.groups)
    # Class centers
    data_class_mean = data_class_grp.mean()

    # Class covariance matrix
    data_class_cov = data_class_grp.cov()

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

        class_ellipses[class_cur] = {"contour": {"x": ell_x, "y": ell_y},
                                     "center": {"x": mean_x, "y": mean_y}}

    return class_ellipses


def plotly_2d_draw_clust_ellipses(
        data_df,
        var_cls,
        var_x=None,
        var_y=None,
        conf_alpha=0.95,
        nb_pts=100,
        class_color_map=None,
        showlegend=False,
        show_clust_center=False,
        fig=None):

    if fig is None:
        fig = go.Figure()

    if var_x is None:
        var_x = data_df.columns[0]
    if var_y is None:
        var_y = data_df.columns[1]

    class_ellipses = \
        compute_2d_clust_ellipses(
            data_df=data_df,
            var_cls=var_cls,
            var_x=var_x,
            var_y=var_y,
            conf_alpha=conf_alpha,
            nb_pts=nb_pts)

    for class_cur, class_ellipse in class_ellipses.items():
        fig.add_scatter(
            legendgroup=class_cur,
            name=f"{class_cur} {conf_alpha:.0%} contour",
            mode="lines",
            line=dict(
                color=class_color_map[class_cur],
                dash="dash",
            ),
            **class_ellipse["contour"])

        if show_clust_center:
            fig.add_scatter(
                legendgroup=class_cur,
                name=f"{class_cur} center",
                mode="markers",
                marker_symbol="x",
                marker_size=12,
                line=dict(color=class_color_map[class_cur]),
                x=[class_ellipse["center"]["x"]],
                y=[class_ellipse["center"]["y"]])

    return fig


def compute_inertia(data_df,
                    point_ref=None,
                    weights=None,
                    dist='euclidean',
                    **dist_params):

    if weights is None:
        weights = pd.Series(1, index=data_df.index)
    if isinstance(weights, (int, float)):
        weights = pd.Series(weights, index=data_df.index)
    # Here we suppose weights is a Pandas Series with consistent index

    if point_ref is None:
        # Compute weighted mean point_ref by default
        point_ref, tmp = compute_weighted_mean(data_df, weights)

    # Compute squared distance between data and point_ref
    dist2_to_point = cdist(data_df,
                           point_ref.to_frame().transpose(),
                           metric=dist,
                           **dist_params)**2

    # Convert numpy vector into Pandas series
    dist2_to_point_s = pd.Series(dist2_to_point.flatten(),
                                 index=data_df.index)

    # Weights distance
    dist2_to_point_weighted_s = dist2_to_point_s*weights

    # Summing to find out inertia
    inertia = dist2_to_point_weighted_s.sum()

    return inertia, point_ref, dist2_to_point_weighted_s.rename("inertia")


def plotly_2d_data_to_point(fig,
                            data_df,
                            point,
                            var_x=None,
                            var_y=None,
                            **line_props):

    if var_x is None:
        var_x = data_df.columns[0]
    if var_y is None:
        var_y = data_df.columns[1]

    var = [var_x, var_y]

    data_bis_df = data_df[var]
    point_bis = point[var]

    _line_props = dict()
    _line_props.update(**line_props)

    x0, y0 = point_bis

    # Create the first line separatly to avoid repeating legend
    x1, y1 = data_bis_df.iloc[0]
    fig.add_scatter(mode="lines",
                    x=[x0, x1],
                    y=[y0, y1],
                    **_line_props)

    for data_idx in data_bis_df.index[1:]:
        x1, y1 = data_bis_df.loc[data_idx]

        fig.add_scatter(mode="lines",
                        x=[x0, x1],
                        y=[y0, y1],
                        showlegend=False,
                        **_line_props)

    return fig


def plotly_2d_highlight_inertia(
        fig,
        data_df,
        point_ref=None,
        var_x=None,
        var_y=None,
        show_data_marker=True,
        show_inertia_line=True,
        show_inertia_point=True,
        annote_inertia=True,
        inertia_params={},
        inertia_text_props={},
        inertia_point_marker_props={},
        data_marker_props={},
        line_props={}):

    if var_x is None:
        var_x = data_df.columns[0]
    if var_y is None:
        var_y = data_df.columns[1]

    inertia, point_ref, tmp = \
        compute_inertia(data_df,
                        point_ref=point_ref,
                        **inertia_params)

    if show_inertia_line:
        # Draw lines between data selection and point_ref
        fig = \
            plotly_2d_data_to_point(fig,
                                    data_df,
                                    point=point_ref,
                                    var_x=var_x,
                                    var_y=var_y,
                                    **line_props)

    if show_inertia_point:

        # Draw and styles point_ref marker
        _inertia_point_marker_props = dict(
            name="Reference point",
            marker_symbol="circle-x",
            marker_size=12,
            marker_line_color=line_props.get("line_color", "black"),
            marker_color=f'rgba{hex_to_rgb(line_props.get("line_color", "black")) + (0.5, )}',
            marker_line_width=2)

        _inertia_point_marker_props.update(**inertia_point_marker_props)

        hover_template = f'{var_x} = %{{x:.2f}}<br>{var_y} = %{{y:.2f}}'
        fig.add_scatter(
            x=[point_ref[var_x]],
            y=[point_ref[var_y]],
            mode="markers",
            hovertemplate=hover_template + f'<br>Inertia = {inertia:.2f}',
            **_inertia_point_marker_props)

    if show_data_marker:
        # Draw and styles data markers
        _data_marker_props = dict(
            name="Data group",
            marker_size=8,
            marker_line_width=2,
            marker_line_color=line_props.get("line_color", "black"))

        _data_marker_props.update(**data_marker_props)

        fig.add_scatter(
            x=data_df[var_x],
            y=data_df[var_y],
            mode="markers",
            hovertemplate=hover_template,
            **_data_marker_props)

    # Compute inertia
    if annote_inertia:

        _inertia_text_props = dict(
            showarrow=False,
            ay=20,
            arrowhead=1,
            arrowwidth=2,
            arrowcolor=_inertia_point_marker_props.get(
                "marker_line_color", "black"),
            font=dict(
                size=10,
                color=_inertia_point_marker_props.get(
                    "marker_line_color", "black")
            ),
            align="center",
            borderpad=4,
            # borderwidth=2,
            # bordercolor=_inertia_point_marker_props.get("marker_line_color", "black"),
            bgcolor="white"
        )

        _inertia_text_props.update(**inertia_text_props)
        # import ipdb
        # ipdb.set_trace()
        fig.add_annotation(
            x=point_ref[var_x],
            y=point_ref[var_y],
            text=f"I={inertia:.2f}",
            **_inertia_text_props
        )

    return fig


def plotly_2d_clust(
        data_df,
        data_weights=None,
        dist="euclidean",
        title=None,
        var_x=None,
        var_y=None,
        var_cls=None,
        var_text=None,
        cls_vlist=None,
        cls_cmap=None,
        between_inertia_color="#777777",
        show_text=False,
        show_cls_center=False,
        show_cls_ellipse=False,
        show_within_inertia_line=False,
        show_between_inertia_line=False,
        data_marker_props={},
        fig=None):

    if fig is None:
        fig = go.Figure()

    if data_weights is None:
        data_weights = pd.Series(1, index=data_df.index)
    if isinstance(data_weights, (int, float)):
        data_weights = pd.Series(data_weights, index=data_df.index)

    if var_x is None:
        var_x = data_df.columns[0]
    if var_y is None:
        var_y = data_df.columns[1]

    data_bis_df = data_df.drop(var_cls, axis=1)

    data_grp_cls = data_bis_df.groupby(data_df[var_cls])

    data_cls_center_df = pd.DataFrame(index=cls_vlist,
                                      columns=data_bis_df.columns)
    data_cls_center_weights = pd.Series(index=cls_vlist)

    dist_params = {}
    if dist == "mahalanobis":
        # Calcul de l'inverse de la matrice de variance des données
        dist_params = \
            {"VI": np.linalg.pinv(data_df.cov().values)}

    for cls, data_cls_df in data_grp_cls:

        scatter_mode = "markers"

        _data_marker_props = dict(
            name=f"{cls} data",
            marker_size=9,
            marker_color=cls_cmap[cls])

        _data_marker_props.update(**data_marker_props)

        data_text = None
        if not(var_text is None):
            if var_text == 'index':
                data_text = data_cls_df.index
            else:
                data_text = data_cls_df[var_text]

        hover_template = f'{var_x} = %{{x:.2f}}<br>{var_y} = %{{y:.2f}}'

        _data_text_props = {}
        if not(data_text is None):
            hover_template = '<b>%{text}</b><br>' + hover_template
            _data_text_props = dict(
                text=data_text,
                textposition="top center",
                textfont_size=_data_marker_props.get("marker_size"),
                textfont_color=cls_cmap[cls],
            )

            if show_text:
                scatter_mode += "+text"

        fig.add_scatter(
            x=data_cls_df[var_x],
            y=data_cls_df[var_y],
            mode=scatter_mode,
            legendgroup=cls,
            hovertemplate=hover_template,
            **_data_text_props,
            **_data_marker_props)

        cls_center, cls_weight = \
            compute_weighted_mean(
                data_df=data_cls_df,
                weights=data_weights.loc[data_cls_df.index])

        data_cls_center_df.loc[cls] = cls_center
        data_cls_center_weights[cls] = cls_weight

        fig = \
            plotly_2d_highlight_inertia(
                data_df=data_cls_df,
                point_ref=cls_center,
                var_x=var_x,
                var_y=var_y,
                fig=fig,
                show_data_marker=False,
                show_inertia_line=show_within_inertia_line,
                show_inertia_point=show_cls_center,
                annote_inertia=False,
                inertia_params=dict(
                    weights=data_weights,
                    dist=dist,
                    **dist_params
                ),
                inertia_point_marker_props=dict(
                    name=f"{cls} center",
                    legendgroup=cls,
                ),
                line_props=dict(
                    name=f"{cls} IW lines",
                    line_color=cls_cmap[cls],
                    legendgroup=cls,
                ))

    # Plot cluster ellipse if required
    if show_cls_ellipse:
        fig = \
            plotly_2d_draw_clust_ellipses(
                data_df=data_df,
                var_cls=var_cls,
                var_x=var_x,
                var_y=var_y,
                class_color_map=cls_cmap,
                showlegend=True,
                show_clust_center=False,
                fig=fig)

    # Plot inertia-between
    if show_between_inertia_line:

        data_center, tmp = \
            compute_weighted_mean(
                data_df=data_bis_df,
                weights=data_weights)

        fig = \
            plotly_2d_highlight_inertia(
                data_df=data_cls_center_df,
                point_ref=data_center,
                var_x=var_x,
                var_y=var_y,
                fig=fig,
                show_data_marker=False,
                show_inertia_line=show_between_inertia_line,
                show_inertia_point=True,
                annote_inertia=False,
                inertia_params=dict(
                    weights=data_cls_center_weights,
                    dist=dist,
                    **dist_params
                ),
                inertia_point_marker_props=dict(
                    name="data center",
                    legendgroup="IB",
                ),
                line_props=dict(
                    name="IB lines",
                    line_color=between_inertia_color,
                    line_width=5,
                    opacity=0.33,
                    legendgroup="IB",
                ))

    fig.update_layout(title=title or "",
                      xaxis_title=var_x,
                      yaxis_title=var_y)

    return fig


def kmeans_algo(data_df, nb_cls=3,
                dist="euclidean",
                cls_init=None,
                iter_max=30,
                var_cls="cls",
                cls_labels=None,
                weights=None,
                return_iter_info=False,
                random_seed=None):

    nb_data = len(data_df)

    if weights is None:
        weights = pd.Series(1, index=data_df.index)
    if isinstance(weights, (int, float)):
        weights = pd.Series(weights, index=data_df.index)

    if cls_labels is None:
        cls_labels = list(range(nb_cls))
    else:
        cls_labels = cls_labels[:nb_cls]

    dist_params = {}
    if dist == "mahalanobis":
        # Calcul de l'inverse de la matrice de variance des données
        dist_params = \
            {"VI": np.linalg.pinv(data_df.cov().values)}

    # Calcul de l'inertie total
    data_it = compute_inertia(data_df,
                              weights=weights,
                              dist=dist,
                              **dist_params)[0]

    # import ipdb
    # ipdb.set_trace()

    # Initialisation
    # itération t=0
    it = 0
    cls_values = []

    # Première partition (aléatoire ou fournie par l'utilisateur)
    if cls_init is None:
        if not(random_seed is None):
            np.random.seed(random_seed)
        cls_init = np.random.choice(nb_cls, nb_data)
        # On s'assure qu'il y ait au moins un individu affecté
        # à chaque classe
        cls_init[:nb_cls] = np.arange(nb_cls)

    cls_values.append(np.array(cls_init))

    # Calcul du centres des classes
    cls_centers = []

    data_cls_grp = data_df.groupby(cls_values[it])
    cls_centers.append(data_cls_grp.apply(lambda d:
                                          compute_weighted_mean(d, weights.loc[d.index])[0]))

    # Inertie intra-classe
    cls_inertia = []
    cls_iw = 0
    for cls, data_cls_df in data_cls_grp:
        cls_iw += \
            compute_inertia(data_cls_df,
                            point_ref=cls_centers[it].loc[cls],
                            weights=weights.loc[data_cls_df.index],
                            dist=dist,
                            **dist_params)[0]
    cls_inertia.append({"IW": cls_iw,
                        "PctI": 1 - cls_iw/data_it})

    stop_criterion = False
    while not(stop_criterion):
        # Iteration t
        it = it + 1

        # Étape 1 : Affectation des classes
        # Calcul des distance aux centres
        dist_cls_centers = cdist(data_df,
                                 cls_centers[it - 1],
                                 metric=dist,
                                 **dist_params)

        # Choix de la classe qui minimise la distance au centre
        cls_values.append(dist_cls_centers.argmin(axis=1))

        # Étape 2 : Mise à jour des centres
        data_cls_grp = data_df.groupby(cls_values[it])

        cls_centers.append(
            data_cls_grp.apply(lambda d: compute_weighted_mean(d, weights.loc[d.index])[0]))

        # Calcul de l'inertie intra classe
        cls_iw = 0
        for cls, data_cls_df in data_cls_grp:
            cls_iw += \
                compute_inertia(data_cls_df,
                                point_ref=cls_centers[it].loc[cls],
                                weights=weights.loc[data_cls_df.index],
                                dist=dist,
                                **dist_params)[0]

        cls_inertia.append({"IW": cls_iw,
                            "PctI": 1 - cls_iw/data_it})

        # ipdb.set_trace()
        stop_cls_unchanged = (cls_values[-1] == cls_values[-2]).all()
        stop_iter_max = it >= iter_max
        stop_criterion = stop_cls_unchanged or stop_iter_max

    # Number of iterations
    nb_iter = len(cls_values)

    # Mise en forme des résultats
    if not(return_iter_info):

        cls_values_final = pd.Series(cls_values[-1],
                                     name=var_cls,
                                     index=data_df.index)\
            .replace({i: cls_labels[i] for i in range(nb_cls)})

        cls_centers_final = cls_centers[-1]
        cls_centers_final.index = cls_labels
        cls_centers_final.index.name = var_cls

        cls_inertia_final = pd.Series(cls_inertia[-1],
                                      name="Inertia")

        return cls_values_final, cls_centers_final, cls_inertia_final

    else:

        cls_values_it = \
            pd.DataFrame(cls_values).stack()

        cls_values_it.replace({i: cls_labels[i]
                               for i in range(nb_cls)},
                              inplace=True)

        cls_values_it.index = \
            pd.MultiIndex.from_product([range(nb_iter), data_df.index],
                                       names=["iter", data_df.index.name])
        cls_values_it.name = var_cls

        cls_centers_it_df = pd.concat(cls_centers)
        cls_centers_it_df.index = \
            pd.MultiIndex.from_product([range(nb_iter),
                                        cls_labels[:nb_cls]],
                                       names=["iter", var_cls])

        cls_inertia_it = \
            pd.DataFrame(cls_inertia, index=range(nb_iter))

        cls_inertia_it.index.name = "iter"

        return cls_values_it, cls_centers_it_df, cls_inertia_it


def plotly_2d_clust_animation(
        data_df,
        cls_values_it,
        **plotly_2d_clust_params):

    # ipdb.set_trace()
    plotly_2d_clust_params.setdefault("var_cls", cls_values_it.loc[0].name)

    data_bis_df = pd.concat([data_df, cls_values_it.loc[0]],
                            axis=1)

    fig_it = plotly_2d_clust(data_bis_df,
                             **plotly_2d_clust_params)

    # ipdb.set_trace()
    fig_dict = {
        "data": fig_it.data,
        "layout": fig_it.layout,
        "frames": []
    }

    fig_dict["layout"]["updatemenus"] = \
        [{
            "buttons":
            [
                {
                    "args": [None, {
                        "frame": {
                            "duration": 1000,
                            "redraw": False
                        },
                        "fromcurrent": True,
                        "transition": {
                            "duration": 0,
                        }
                    }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                    ],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    for it in cls_values_it.index.levels[0]:
        data_bis_df = pd.concat([data_df, cls_values_it.loc[it]],
                                axis=1)

        fig_it = plotly_2d_clust(data_bis_df,
                                 **plotly_2d_clust_params)

        frame = {"data": fig_it.data,
                 "name": str(it)}

        fig_dict["frames"].append(frame)

        slider_step = {"args": [
            [it],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
            "label": it,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    return fig_dict
