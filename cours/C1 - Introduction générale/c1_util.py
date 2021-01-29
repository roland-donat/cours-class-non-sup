import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform

def compute_inertia(data_df, point=None, weights=1, dist='euclidean'):
    
    if weights is None:
        weights = pd.Series(1/len(data_df), index=data_df.index)
        
    if point is None:
        # Compute weighted mean point by default
        point = (data_df * weights).sum()/weights.sum()
        
    # Compute squared distance between data and point
    dist2_to_point = cdist(data_df,
                           point.to_frame().transpose(),
                           metric=dist)**2
    
    # Convert numpy vector into Pandas series
    dist2_to_point_s = pd.Series(dist2_to_point.flatten(),
                                 index=data_df.index)
    
    # Weights distance
    dist2_to_point_weighted_s = dist2_to_point_s*weights
    
    # Summing to find out inertia
    inertia = dist2_to_point_weighted_s.sum()
    
    return inertia, point, dist2_to_point_weighted_s.rename("inertia")

def plotly_2d_data_to_point(fig,
                            data_df, 
                            point,
                            var=None,
                            **line_props):
    
    if var is None:
        data_sel_df = data_df.iloc[:, :2]
        point_sel = point.iloc[:2]
        
    else:
        data_sel_df = data_df[var]
        point_sel = point[var]
    
    x0, y0 = point_sel
    for data_idx in data_sel_df.index:
        x1, y1 = data_sel_df.loc[data_idx]
        
        # fig.add_shape(type="line", 
        #               x0=x0, y0=y0, x1=x1, y1=y1,
        #               **line_props)
        fig.add_scatter(mode="lines", 
                        x=[x0, x1],
                        y=[y0, y1],
                        showlegend=False,
                        **line_props)
        
    return fig


def plotly_2d_highlight_inertia(
        fig,
        data_df, 
        point,
        var=None,
        annote_inertia=True,
        inertia_params={},
        inertia_text_props={},
        point_marker_props={},
        data_marker_props={},
        line_props={}):
    
    if var is None:
        data_sel_df = data_df.iloc[:, :2]
        point_sel = point.iloc[:2]
        
    else:
        data_sel_df = data_df[var]
        point_sel = point[var]
        
    var_x = data_sel_df.columns[0]
    var_y = data_sel_df.columns[1]
    
    inertia, _, _ = \
        compute_inertia(data_sel_df, 
                        point=point, 
                        **inertia_params)
    
    # Draw lines between data selection and point
    fig = \
        plotly_2d_data_to_point(fig,
                                data_df, 
                                point=point,
                                **line_props)
    
    # Draw and styles point marker
    _point_marker_props = dict(
        name="Reference point",
        marker_symbol="circle-x",
        marker_size=12,
        marker_line_color=line_props.get("line_color", "black"),
        marker_line_width=2)
    
    _point_marker_props.update(**point_marker_props)
    
    hover_template = f'{var_x} = %{{x:.2f}}<br>{var_y} = %{{y:.2f}}'
    fig.add_scatter(
        x=[point.iloc[0]],
        y=[point.iloc[1]],
        mode="markers",
        hovertemplate=hover_template + f'<br>Inertia = {inertia:.2f}',
        **_point_marker_props)
    
    # Draw and styles data markers
    _data_marker_props = dict(
        name="Data group",
        marker_size=8,
        marker_line_width=2,
        marker_line_color=line_props.get("line_color", "black"))
    
    _data_marker_props.update(**data_marker_props)
    
    fig.add_scatter(
        x=data_sel_df.iloc[:,0],
        y=data_sel_df.iloc[:,1],
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
            arrowcolor=_point_marker_props.get("marker_line_color", "black"),
            font=dict(
                size=10,
                color=_point_marker_props.get("marker_line_color", "black")
            ),
            align="center",
            borderpad=4,
            # borderwidth=2,
            # bordercolor=_point_marker_props.get("marker_line_color", "black"),
            bgcolor="white"
        )
        
        _inertia_text_props.update(**inertia_text_props)
        # import ipdb
        # ipdb.set_trace()
        fig.add_annotation(
            x=point.iloc[0], 
            y=point.iloc[1],
            text=f"I={inertia:.2f}",
            **_inertia_text_props
        )
        
        
    return fig
