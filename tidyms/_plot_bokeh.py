import numpy as np
import bokeh.plotting
from bokeh.palettes import Set3
from .peaks import Peak
from typing import List, Optional


def plot_chromatogram(rt: np.ndarray, spint: np.ndarray,
                      peaks: Optional[List[Peak]] = None,
                      draw: bool = True, fig_params: Optional[dict] = None,
                      line_params: Optional[dict] = None
                      ) -> bokeh.plotting.Figure:
    """
    Plots a chromatogram

    Parameters
    ----------
    rt : array
        array of retention time
    spint : array
        array of intensity
    peaks : List[Peaks]
    draw : bool, optional
        if True run bokeh show function.
    fig_params : dict
        key-value parameters to pass into bokeh figure function.
    line_params : dict
        key-value parameters to pass into bokeh line function.

    Returns
    -------
    bokeh Figure

    """
    default_line_params = {"line_width": 1, "line_color": "black", "alpha": 0.8}
    default_fig_params = {"aspect_ratio": 1.5}
    cmap = Set3[12]

    if line_params is None:
        line_params = default_line_params
    else:
        for params in line_params:
            default_line_params[params] = line_params[params]
        line_params = default_line_params

    if fig_params is None:
        fig_params = default_fig_params
    else:
        default_fig_params.update(fig_params)
        fig_params = default_fig_params

    fig = bokeh.plotting.figure(**fig_params)
    fig.line(rt, spint, **line_params)
    if peaks is not None:
        for k, peak in enumerate(peaks):
            fig.varea(rt[peak.start:peak.end], spint[peak.start:peak.end],
                      0, fill_alpha=0.8, fill_color=cmap[k % 12])
            # k % 12 is used to cycle over the colormap

    #  figure appearance
    fig.xaxis.axis_label = "Rt [s]"
    fig.yaxis.axis_label = "intensity [au]"
    fig.yaxis.axis_label_text_font_style = "bold"
    fig.yaxis.formatter.precision = 2
    fig.xaxis.axis_label_text_font_style = "bold"

    if draw:
        bokeh.plotting.show(fig)
    return fig
